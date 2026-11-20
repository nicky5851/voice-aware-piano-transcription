"""Thin wrapper around the vendored RMVPE inference code (spec WP1 §4.4.2).

Stores f0_hz and salience (max of the hidden salience) separately, per the
spec's output contract, rather than relying on the vendored decode()'s own
built-in threshold -- voicing is decided by our own postprocess_voicing().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from vapt.data.pitch import PitchTrack

THIRD_PARTY_RMVPE = Path(__file__).resolve().parents[3] / "third_party" / "rmvpe"

RMVPE_VOICED_THRESHOLD = 0.03  # spec §4.4.2: "0.03 in the common implementation"
RMVPE_HOP_S = 0.01  # the vendored MelSpectrogram is constructed with hop_length=160 @ 16kHz


def _ensure_vendored_on_path() -> None:
    p = str(THIRD_PARTY_RMVPE)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_rmvpe_model(checkpoint_path: Path, device: str = "cuda", is_half: bool = False):
    _ensure_vendored_on_path()
    from rmvpe import RMVPE  # vendored

    return RMVPE(str(checkpoint_path), is_half=is_half, device=device)


def run_rmvpe(audio_16k: np.ndarray, model) -> PitchTrack:
    """audio_16k: mono float32 [-1, 1] at 16kHz. Returns an unvoiced-thresholded-by-us PitchTrack."""
    mel = model.extract_mel(audio_16k, center=True)
    hidden = model.mel2hidden(mel)
    if "privateuseone" not in str(model.device):
        hidden_np = hidden.squeeze(0).cpu().numpy()
    else:
        hidden_np = hidden[0]
    if model.is_half:
        hidden_np = hidden_np.astype("float32")

    salience = hidden_np.max(axis=1).astype(np.float32)

    # thred=0.0: get the raw f0 at every frame; we threshold voicing ourselves.
    cents = model.to_local_average_cents(hidden_np, thred=0.0)
    f0 = (10.0 * (2.0 ** (cents / 1200.0))).astype(np.float32)
    f0[f0 == 10.0] = 0.0  # the vendored decode()'s own sentinel for "no pitch"

    voiced = salience >= RMVPE_VOICED_THRESHOLD
    return PitchTrack(f0_hz=f0, salience=salience, voiced=voiced, source="rmvpe", hop_s=RMVPE_HOP_S)
