"""torchcrepe wrapper: the CREPE cross-check and CREPE fallback (spec WP1 §4.4.4, §4.4.5)."""
from __future__ import annotations

import numpy as np
import torch
import torchcrepe

from vapt.data.pitch import PitchTrack

CREPE_HOP_S = 0.01  # hop=160 @ 16kHz
CREPE_FMIN = 50.0
CREPE_FMAX = 1100.0
CREPE_VOICED_THRESHOLD = 0.21  # torchcrepe's own documented default periodicity-silence split


def run_crepe(audio_16k: np.ndarray, device: str = "cpu") -> PitchTrack:
    """audio_16k: mono float32 [-1, 1] at 16kHz. "full" model, Viterbi decoding, hop 160."""
    audio_t = torch.from_numpy(audio_16k).float().unsqueeze(0)
    f0, periodicity = torchcrepe.predict(
        audio_t,
        sample_rate=16000,
        hop_length=160,
        fmin=CREPE_FMIN,
        fmax=CREPE_FMAX,
        model="full",
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
        device=device,
        batch_size=512,
    )
    f0 = f0.squeeze(0).cpu().numpy().astype(np.float32)
    salience = periodicity.squeeze(0).cpu().numpy().astype(np.float32)
    voiced = salience >= CREPE_VOICED_THRESHOLD
    return PitchTrack(f0_hz=f0, salience=salience, voiced=voiced, source="crepe", hop_s=CREPE_HOP_S)


def cents_diff(f0_a: np.ndarray, f0_b: np.ndarray) -> np.ndarray:
    """|cents(a) - cents(b)|, undefined (nan) where either is non-positive."""
    valid = (f0_a > 0) & (f0_b > 0)
    out = np.full(f0_a.shape, np.nan, dtype=np.float64)
    out[valid] = np.abs(1200.0 * np.log2(f0_a[valid] / f0_b[valid]))
    return out


def jointly_voiced_agreement(track_a: PitchTrack, track_b: PitchTrack, cents_tol: float = 50.0) -> float:
    """Share of frames voiced by both tracks whose f0 agrees within cents_tol (spec §4.4.4)."""
    both_voiced = track_a.voiced & track_b.voiced
    if not np.any(both_voiced):
        return float("nan")
    diff = cents_diff(track_a.f0_hz[both_voiced], track_b.f0_hz[both_voiced])
    return float(np.mean(diff <= cents_tol))


def voicing_cohen_kappa(voiced_a: np.ndarray, voiced_b: np.ndarray) -> float:
    """Cohen's kappa for voicing agreement (spec §4.4.4)."""
    n = voiced_a.shape[0]
    if n == 0:
        return float("nan")
    observed = np.mean(voiced_a == voiced_b)
    p_a = np.mean(voiced_a)
    p_b = np.mean(voiced_b)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected >= 1.0:
        return 1.0
    return float((observed - expected) / (1.0 - expected))
