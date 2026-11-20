"""Validation-only evaluation entry point, used by training (spec WP2 SS8, WP3 SS3.6).

Refuses manifest="test": only src/vapt/eval/final.py (WP6) may score test
ground truth. Reads ground-truth MIDI and voice pitch tracks for the VAL
split only -- the isolation rule (tests/test_isolation.py) restricts the
TEST split's MIDI, not val's.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from vapt.constants import BEGIN_NOTE, CLASSES_NUM, FRAMES_PER_SECOND
from vapt.data.pitch import PitchTrack, load_pitch
from vapt.eval.melody import voice_metrics
from vapt.eval.metrics import score_clip
from vapt.mixing.manifests import render

THIRD_PARTY_UTILS = Path(__file__).resolve().parents[3] / "third_party" / "piano_transcription" / "utils"


def _ensure_vendored_on_path() -> None:
    p = str(THIRD_PARTY_UTILS)
    if p not in sys.path:
        sys.path.insert(0, p)


def ground_truth_notes(hdf5s_dir: Path, entry: dict) -> list[dict]:
    """Note events for one manifest entry's piano window, clip-relative times."""
    _ensure_vendored_on_path()
    from utilities import TargetProcessor

    path = Path(hdf5s_dir) / entry["maestro_year"] / entry["maestro_file"]
    with h5py.File(path, "r") as hf:
        midi_events = [e.decode() for e in hf["midi_event"][:]]
        midi_events_time = hf["midi_event_time"][:]

    tp = TargetProcessor(entry["dur_s"], FRAMES_PER_SECOND, BEGIN_NOTE, CLASSES_NUM)
    start = entry["piano_start_s"]
    dur_s = entry["dur_s"]
    _target_dict, note_events, _pedal_events = tp.process(start, midi_events_time, midi_events, extend_pedal=True)

    # ENGINEERING CHOICE: TargetProcessor's cross-segment backtracking (ex_bgn_idx) can return
    # notes that started before the window (held into it, e.g. via pedal); once shifted by
    # `start` these have a negative onset, which mir_eval.transcription rejects. Clip into
    # [0, dur_s] and drop notes entirely outside, matching TargetProcessor's own frame-clamping
    # when it paints such notes into the rolls -- no scientific semantics change.
    out = []
    for n in note_events:
        onset = n["onset_time"] - start
        offset = n["offset_time"] - start
        if offset <= 0.0 or onset >= dur_s:
            continue
        out.append(
            {
                "midi_note": n["midi_note"],
                "onset_time": max(onset, 0.0),  # clip notes held over from before the window, matching
                "offset_time": min(offset, dur_s),  # TargetProcessor's own frame-clamping for the rolls
                "velocity": n["velocity"],
            }
        )
    return out


def ground_truth_voice_track(pitch_dir: Path, entry: dict, musdb_split: str, n_frames: int) -> PitchTrack:
    """The reference voice pitch track for one manifest entry's window, sized to n_frames."""
    if entry["condition"] == "clean":
        return PitchTrack(
            f0_hz=np.zeros(n_frames, dtype=np.float32),
            salience=np.zeros(n_frames, dtype=np.float32),
            voiced=np.zeros(n_frames, dtype=bool),
            source="clean",
            hop_s=1.0 / FRAMES_PER_SECOND,
        )

    track = load_pitch(Path(pitch_dir) / musdb_split / f"{entry['musdb_song']}.npz")
    start_frame = int(round(entry["voice_start_s"] * FRAMES_PER_SECOND))
    f0 = np.zeros(n_frames, dtype=np.float32)
    salience = np.zeros(n_frames, dtype=np.float32)
    voiced = np.zeros(n_frames, dtype=bool)

    src_start = max(start_frame, 0)
    src_end = min(start_frame + n_frames, track.f0_hz.shape[0])
    if src_start < src_end:
        dst_start = src_start - start_frame
        dst_end = dst_start + (src_end - src_start)
        f0[dst_start:dst_end] = track.f0_hz[src_start:src_end]
        salience[dst_start:dst_end] = track.salience[src_start:src_end]
        voiced[dst_start:dst_end] = track.voiced[src_start:src_end]

    return PitchTrack(f0_hz=f0, salience=salience, voiced=voiced, source=track.source, hop_s=track.hop_s)


def score_item(note_events: list[dict], voice_roll_pred: np.ndarray, ref_notes: list[dict], ref_track: PitchTrack, dur_s: float) -> dict:
    """Score one clip's piano notes and voice-head output against ground truth."""
    assert voice_roll_pred.shape[0] == ref_track.f0_hz.shape[0], (
        f"predicted voice roll has {voice_roll_pred.shape[0]} frames, "
        f"ground-truth track has {ref_track.f0_hz.shape[0]}: a framing bug, not recoverable here"
    )
    cs = score_clip(ref_notes, note_events, ref_track if ref_track.voiced.any() else None, dur_s)
    vm = voice_metrics(voice_roll_pred, ref_track)
    out = asdict(cs)
    out.update({f"voice_{k}": v for k, v in vm.items()})
    return out


def score_manifest(
    transcribe_fn,
    entries: list[dict],
    hdf5s_dir: Path,
    vocals16k_dir: Path,
    pitch_dir: Path,
    musdb_split: str = "val",
) -> list[dict]:
    """Render each manifest entry, transcribe it with `transcribe_fn(audio) -> (notes, voice_roll)`, and score it.

    Used directly by train.py's validation step (in-process, no disk round
    trip) and by evaluate_predictions below (reading saved predictions back).
    """
    rows = []
    for entry in entries:
        audio, _info = render(entry, hdf5s_dir, vocals16k_dir, musdb_split=musdb_split)
        note_events, voice_roll_pred = transcribe_fn(audio)
        ref_notes = ground_truth_notes(hdf5s_dir, entry)
        ref_track = ground_truth_voice_track(pitch_dir, entry, musdb_split, voice_roll_pred.shape[0])
        row = {"clip_id": entry["clip_id"], "condition": entry["condition"]}
        row.update(score_item(note_events, voice_roll_pred, ref_notes, ref_track, entry["dur_s"]))
        rows.append(row)
    return rows


def evaluate_predictions(
    pred_dir: Path,
    manifest: str = "val",
    manifest_path: Path = Path("manifests/val_manifest.json"),
    hdf5s_dir: Path = Path("/workspace/vapt/work/hdf5s/maestro"),
    pitch_dir: Path = Path("/workspace/vapt/work/pitch"),
    musdb_split: str = "val",
) -> dict:
    """Score predictions already written to disk by infer.py's CLI (standalone use, e.g. WP4's evaluate.py)."""
    if manifest == "test":
        raise ValueError(
            "evaluate_predictions refuses manifest='test': test scoring happens only in "
            "src/vapt/eval/final.py (WP6), per protocol SS9.7"
        )
    import pretty_midi

    entries = json.loads(Path(manifest_path).read_text())
    rows = []
    for entry in entries:
        mid_path = Path(pred_dir) / f"{entry['clip_id']}.mid"
        pm = pretty_midi.PrettyMIDI(str(mid_path))
        note_events = [
            {"midi_note": note.pitch, "onset_time": note.start, "offset_time": note.end, "velocity": note.velocity}
            for inst in pm.instruments
            for note in inst.notes
        ]
        voice_roll_pred = np.load(Path(pred_dir) / f"{entry['clip_id']}_voice.npz")["voice_roll"]
        ref_notes = ground_truth_notes(hdf5s_dir, entry)
        ref_track = ground_truth_voice_track(pitch_dir, entry, musdb_split, voice_roll_pred.shape[0])
        row = {"clip_id": entry["clip_id"], "condition": entry["condition"]}
        row.update(score_item(note_events, voice_roll_pred, ref_notes, ref_track, entry["dur_s"]))
        rows.append(row)
    return {"rows": rows}
