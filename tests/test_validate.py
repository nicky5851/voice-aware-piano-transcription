import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from vapt.data.pitch import PitchTrack, save_pitch
from vapt.eval.validate import (
    evaluate_predictions,
    ground_truth_notes,
    ground_truth_voice_track,
    score_item,
    score_manifest,
)

FPS = 100


def _write_h5(path: Path, year: str, split: str, duration: float, midi_events: list[str], midi_events_time: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        hf.attrs["year"] = year
        hf.attrs["split"] = split
        hf.attrs["duration"] = duration
        hf.create_dataset("midi_event", data=[e.encode() for e in midi_events])
        hf.create_dataset("midi_event_time", data=np.array(midi_events_time, dtype=np.float64))


def test_ground_truth_notes_shifted_to_clip_relative_time(tmp_path):
    hdf5s_dir = tmp_path / "hdf5s"
    events = [
        "note_on channel=0 note=60 velocity=80 time=0",
        "note_off channel=0 note=60 velocity=0 time=0",
        "note_on channel=0 note=61 velocity=80 time=0",  # past the segment end, just so
        "note_off channel=0 note=61 velocity=0 time=0",  # TargetProcessor's fin_idx search has something to break on
    ]
    times = [10.5, 10.9, 100.0, 100.1]
    _write_h5(hdf5s_dir / "2004" / "piece.h5", "2004", "validation", 200.0, events, times)

    entry = {"maestro_year": "2004", "maestro_file": "piece.h5", "piano_start_s": 10.0, "dur_s": 30.0}
    notes = ground_truth_notes(hdf5s_dir, entry)
    assert len(notes) == 1
    assert notes[0]["midi_note"] == 60
    assert notes[0]["onset_time"] == pytest.approx(0.5, abs=1e-6)
    assert notes[0]["offset_time"] == pytest.approx(0.9, abs=1e-6)


def test_ground_truth_voice_track_clean_condition_is_all_unvoiced():
    entry = {"condition": "clean"}
    track = ground_truth_voice_track(Path("unused"), entry, "val", n_frames=50)
    assert track.f0_hz.shape == (50,)
    assert not track.voiced.any()


def test_ground_truth_voice_track_windows_and_pads(tmp_path):
    pitch_dir = tmp_path / "pitch"
    (pitch_dir / "val").mkdir(parents=True)
    t_total = 20
    f0 = np.arange(t_total, dtype=np.float32) + 100.0
    voiced = np.ones(t_total, dtype=bool)
    save_pitch(pitch_dir / "val" / "song.npz", PitchTrack(f0_hz=f0, salience=np.ones(t_total, dtype=np.float32), voiced=voiced, source="test"))

    entry = {"condition": "+6", "musdb_song": "song", "voice_start_s": 0.1}  # start_frame = 10
    track = ground_truth_voice_track(pitch_dir, entry, "val", n_frames=15)
    assert track.f0_hz.shape == (15,)
    # frames 10..19 of the source map to dst 0..9; dst 10..14 are padded (unvoiced, f0=0)
    assert np.array_equal(track.f0_hz[:10], f0[10:20])
    assert not track.voiced[10:].any()


def test_score_item_perfect_prediction_gives_f1_100():
    ref_notes = [{"midi_note": 60, "onset_time": 1.0, "offset_time": 1.5, "velocity": 80}]
    n_frames = 300
    track = PitchTrack(
        f0_hz=np.zeros(n_frames, dtype=np.float32),
        salience=np.zeros(n_frames, dtype=np.float32),
        voiced=np.zeros(n_frames, dtype=bool),
        source="test",
    )
    voice_roll_pred = np.zeros((n_frames, 88), dtype=np.float32)
    result = score_item(ref_notes, voice_roll_pred, ref_notes, track, dur_s=3.0)
    assert result["onset_f1"] == pytest.approx(100.0)
    assert result["n_leak"] == 0


def test_score_item_asserts_on_frame_count_mismatch():
    ref_notes: list[dict] = []
    track = PitchTrack(
        f0_hz=np.zeros(10, dtype=np.float32), salience=np.zeros(10, dtype=np.float32), voiced=np.zeros(10, dtype=bool), source="test"
    )
    voice_roll_pred = np.zeros((11, 88), dtype=np.float32)
    with pytest.raises(AssertionError):
        score_item([], voice_roll_pred, ref_notes, track, dur_s=1.0)


WORKSPACE = Path("/workspace/vapt/work")
_HAS_REAL_DATA = (WORKSPACE / "hdf5s" / "maestro").exists() and (WORKSPACE / "pitch" / "val" / "voiced_windows_30s.json").exists()


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="requires WP1 data on the pod's network volume")
def test_score_manifest_runs_against_real_val_manifest():
    entries = json.loads(Path("manifests/val_manifest.json").read_text())[:2]

    def fake_transcribe(audio):
        n_frames = int(round(entries[0]["dur_s"] * FPS)) + 1
        return [], np.zeros((n_frames, 88), dtype=np.float32)

    rows = score_manifest(fake_transcribe, entries, WORKSPACE / "hdf5s" / "maestro", WORKSPACE / "vocals16k", WORKSPACE / "pitch", musdb_split="val")
    assert len(rows) == 2
    for row in rows:
        assert "onset_f1" in row
        assert "voice_rpa" in row


def test_evaluate_predictions_refuses_test_manifest():
    with pytest.raises(ValueError):
        evaluate_predictions(Path("unused"), manifest="test")
