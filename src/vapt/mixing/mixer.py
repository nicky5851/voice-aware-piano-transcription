"""The loudness-controlled mixer (protocol §5 VPR definition, spec WP2 §2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyloudnorm

from vapt.constants import SAMPLE_RATE

MIN_SEGMENT_S = 0.4  # BS.1770 gating block length; pyloudnorm cannot meter below this
PEAK_LIMIT = 0.99


class SilentSegmentError(Exception):
    """Raised when a piano or voice segment cannot be metered (too short or -inf LUFS)."""


@dataclass
class MixInfo:
    vpr_target: float
    vpr_measured: float
    gain_voice: float
    peak_scale: float
    lufs_piano: float
    lufs_voice: float


def _integrated_loudness(audio: np.ndarray, meter: pyloudnorm.Meter) -> float:
    duration_s = audio.shape[0] / SAMPLE_RATE
    if duration_s < MIN_SEGMENT_S:
        raise SilentSegmentError(f"segment is {duration_s:.3f}s, shorter than the {MIN_SEGMENT_S}s gating block")
    lufs = meter.integrated_loudness(audio.astype(np.float64))
    if np.isinf(lufs):
        raise SilentSegmentError("segment has -inf integrated loudness (silence)")
    return float(lufs)


def mix(piano: np.ndarray, voice: np.ndarray, vpr_db: float, meter: pyloudnorm.Meter) -> tuple[np.ndarray, MixInfo]:
    """Mix mono float32 piano and voice at 16kHz to a target voice-to-piano ratio (protocol §5).

    Raises SilentSegmentError if either segment cannot be metered (too short, or -inf LUFS).
    """
    if piano.shape[0] != voice.shape[0]:
        raise ValueError(f"piano and voice must have equal length, got {piano.shape[0]} and {voice.shape[0]}")

    lufs_piano = _integrated_loudness(piano, meter)
    lufs_voice = _integrated_loudness(voice, meter)

    gain_voice = 10.0 ** ((lufs_piano + vpr_db - lufs_voice) / 20.0)
    scaled_voice = gain_voice * voice
    x = piano + scaled_voice

    vpr_measured = _integrated_loudness(scaled_voice, meter) - lufs_piano

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    peak_scale = 1.0
    if peak > PEAK_LIMIT:
        peak_scale = PEAK_LIMIT / peak
        x = x * peak_scale

    info = MixInfo(
        vpr_target=float(vpr_db),
        vpr_measured=float(vpr_measured),
        gain_voice=float(gain_voice),
        peak_scale=float(peak_scale),
        lufs_piano=float(lufs_piano),
        lufs_voice=float(lufs_voice),
    )
    return x.astype(np.float32), info
