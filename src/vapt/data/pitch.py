"""Pitch-track storage and voicing post-processing (protocol §6.1, spec WP1 §4.4)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PitchTrack:
    f0_hz: np.ndarray
    salience: np.ndarray
    voiced: np.ndarray
    source: str
    hop_s: float = 0.01


def load_pitch(path: Path) -> PitchTrack:
    data = np.load(path)
    return PitchTrack(
        f0_hz=data["f0_hz"].astype(np.float32),
        salience=data["salience"].astype(np.float32),
        voiced=data["voiced"].astype(bool),
        source=str(data["source"]),
        hop_s=float(data["hop_s"]),
    )


def save_pitch(path: Path, track: PitchTrack) -> None:
    np.savez(
        path,
        f0_hz=track.f0_hz.astype(np.float32),
        salience=track.salience.astype(np.float32),
        voiced=track.voiced.astype(bool),
        source=track.source,
        hop_s=np.float32(track.hop_s),
    )


def _remove_short_voiced_runs(voiced: np.ndarray, min_run: int) -> np.ndarray:
    out = voiced.copy()
    t_len = out.shape[0]
    i = 0
    while i < t_len:
        if out[i]:
            j = i
            while j < t_len and out[j]:
                j += 1
            if (j - i) < min_run:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def _frame_rms_dbfs(audio: np.ndarray, hop_samples: int, win_samples: int, n_frames: int) -> np.ndarray:
    """25ms-window RMS (dBFS) centred on each 10ms-hop frame, vectorised via a cumsum."""
    sq = audio.astype(np.float64) ** 2
    cumsum = np.concatenate([[0.0], np.cumsum(sq)])
    centers = np.arange(n_frames) * hop_samples
    half = win_samples // 2
    starts = np.clip(centers - half, 0, audio.shape[0])
    ends = np.clip(centers + half, 0, audio.shape[0])
    counts = np.maximum(ends - starts, 1)
    sums = cumsum[ends] - cumsum[starts]
    rms = np.sqrt(sums / counts)
    return 20.0 * np.log10(rms + 1e-12)


def postprocess_voicing(
    track: PitchTrack,
    audio_16k: np.ndarray,
    min_run: int = 5,
    floor_dbfs: float = -60.0,
    sample_rate: int = 16000,
) -> PitchTrack:
    """Remove short voiced runs, then force unvoiced where the stem is too quiet (spec §4.3)."""
    voiced = _remove_short_voiced_runs(track.voiced, min_run)

    t_len = voiced.shape[0]
    hop_samples = int(round(track.hop_s * sample_rate))
    win_samples = int(round(0.025 * sample_rate))
    rms_db = _frame_rms_dbfs(audio_16k, hop_samples, win_samples, t_len)
    voiced = voiced & (rms_db >= floor_dbfs)

    return PitchTrack(
        f0_hz=track.f0_hz, salience=track.salience, voiced=voiced, source=track.source, hop_s=track.hop_s
    )
