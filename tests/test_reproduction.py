"""GPU-only reproduction and sanity checks, run before any training (spec WP3 §5).

These need a CUDA GPU, the real Kong-released checkpoint, and the
piano_transcription_inference pip package (a reproduction-check-only
dependency, never a runtime one -- protocol §10.4/CLAUDE.md §7). They are
excluded from the default CPU test run and only run on a GPU pod at a gate.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from vapt.constants import CLASSES_NUM, FRAMES_PER_SECOND, SAMPLE_RATE
from vapt.infer import SEGMENT_SAMPLES, deframe, enframe, transcribe
from vapt.models.voice_aware import VoiceAwareKong, load_pretrained

WORKSPACE = Path(os.environ.get("VAPT_WORKSPACE", "/workspace/vapt/work"))
CHECKPOINT = Path(
    os.environ.get(
        "VAPT_KONG_CHECKPOINT",
        str(WORKSPACE / "checkpoints" / "kong_note_F1_0.9677_pedal_F1_0.9186.pth"),
    )
)
VAL_MANIFEST = Path(os.environ.get("VAPT_VAL_MANIFEST", "manifests/val_manifest.json"))

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
    pytest.mark.skipif(not CHECKPOINT.exists(), reason="requires the downloaded Kong checkpoint"),
]

DEVICE = "cuda"


def _piano_window() -> np.ndarray:
    """One MAESTRO validation-split 10s window, audio only (isolation-safe: no MIDI read)."""
    from vapt.mixing.manifests import _load_piano_window

    return _load_piano_window(
        WORKSPACE / "hdf5s" / "maestro",
        "2004",
        "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.h5",
        start_s=151.2391154736453,
        dur_s=10.0,
    )


def _our_rolls(model, audio: np.ndarray, lam: float | None) -> dict[str, np.ndarray]:
    """Segment-and-stitch audio through `model`, returning deframed rolls at every output key.

    lam=None calls the vendored parent class's forward (no lam kwarg); a float calls
    VoiceAwareKong's forward(x, lam=...).
    """
    was_training = model.training
    model.eval()
    try:
        audio = audio.astype(np.float32)[None, :]
        audio_len = audio.shape[1]
        pad_len = int(np.ceil(audio_len / SEGMENT_SAMPLES)) * SEGMENT_SAMPLES - audio_len
        audio = np.concatenate([audio, np.zeros((1, pad_len), dtype=np.float32)], axis=1)
        segments = enframe(audio, SEGMENT_SAMPLES)
        collected: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for seg in segments:
                x = torch.from_numpy(seg[None, :]).to(DEVICE)
                out = model(x) if lam is None else model(x, lam=lam)
                for k, v in out.items():
                    collected.setdefault(k, []).append(v.cpu().numpy()[0])
        return {k: deframe(np.stack(v, axis=0))[0:audio_len] for k, v in collected.items()}
    finally:
        model.train(was_training)


def _load_parent_only(path: Path):
    """A bare vendored Regress_onset_offset_frame_velocity_CRNN with Kong's released weights,
    loaded the same way load_pretrained() does for VoiceAwareKong (strip the note_model. prefix)."""
    import sys

    third_party_utils = Path(__file__).resolve().parents[1] / "third_party" / "piano_transcription" / "utils"
    p = str(third_party_utils)
    if p not in sys.path:
        sys.path.insert(0, p)
    from models import Regress_onset_offset_frame_velocity_CRNN

    model = Regress_onset_offset_frame_velocity_CRNN(FRAMES_PER_SECOND, CLASSES_NUM).to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state = checkpoint["model"]["note_model"]
    result = model.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys
    model.eval()
    return model


@pytest.fixture(scope="module")
def pretrained_model():
    model = VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM).to(DEVICE)
    load_pretrained(model, CHECKPOINT)
    model.eval()
    return model


def test_exact_reproduction_matches_pip_package(pretrained_model):
    """WP3 §5.1: on one val MAESTRO window, we must match piano_transcription_inference to <1e-4."""
    from piano_transcription_inference import PianoTranscription

    audio = _piano_window()
    ours = _our_rolls(pretrained_model, audio, lam=1.0)

    pt = PianoTranscription(device=DEVICE, checkpoint_path=str(CHECKPOINT))
    with tempfile.TemporaryDirectory() as td:
        theirs = pt.transcribe(audio, os.path.join(td, "out.mid"))
    theirs_frame = theirs["output_dict"]["frame_output"]

    n = min(ours["frame_output"].shape[0], theirs_frame.shape[0])
    max_abs_diff = float(np.max(np.abs(ours["frame_output"][:n] - theirs_frame[:n])))
    assert max_abs_diff < 1e-4, f"max abs diff {max_abs_diff} >= 1e-4"


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_piano_path_unchanged_for_every_lambda(pretrained_model, lam):
    """WP3 §5.2: VoiceAwareKong's piano outputs must equal the parent class's exactly (eval mode)."""
    audio = _piano_window()
    parent = _load_parent_only(CHECKPOINT)
    parent_out = _our_rolls(parent, audio, lam=None)  # vendored parent's forward takes no lam kwarg
    ours = _our_rolls(pretrained_model, audio, lam=lam)

    piano_keys = ["frame_output", "reg_onset_output", "reg_offset_output", "velocity_output"]
    for key in piano_keys:
        assert key in parent_out and key in ours
        max_abs_diff = float(np.max(np.abs(ours[key] - parent_out[key])))
        assert max_abs_diff == 0.0, f"{key} differs by {max_abs_diff} at lam={lam}"


def test_pretrained_validation_sanity_onset_f1(pretrained_model):
    """WP3 §5.3: pretrained clean onset F1 on the validation manifest must be >= 85."""
    from vapt.eval.validate import score_manifest

    entries = json.loads(VAL_MANIFEST.read_text())
    clean_entries = [e for e in entries if e["condition"] == "clean"]
    assert clean_entries, "no clean-condition entries in the validation manifest"

    def _transcribe_fn(audio: np.ndarray):
        return transcribe(pretrained_model, audio, device=DEVICE)

    results = score_manifest(
        _transcribe_fn,
        clean_entries,
        hdf5s_dir=WORKSPACE / "hdf5s" / "maestro",
        vocals16k_dir=WORKSPACE / "vocals16k",
        pitch_dir=WORKSPACE / "pitch",
        musdb_split="val",
    )
    mean_onset_f1 = float(np.mean([r["onset_f1"] for r in results]))
    assert mean_onset_f1 >= 85.0, f"pretrained clean onset F1 {mean_onset_f1:.2f} < 85 -- pipeline bug, stop"
