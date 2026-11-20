"""The voice_roll target function (protocol §6, spec WP1 §4.5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vapt.constants import BEGIN_NOTE, CLASSES_NUM
from vapt.data.pitch import load_pitch


def voice_roll(
    f0_hz: np.ndarray, voiced: np.ndarray, sigma: float = 0.5, floor: float = 0.01
) -> np.ndarray:
    """[T], [T] -> [T, 88]. Gaussian bump around the sung MIDI note per frame."""
    t_len = f0_hz.shape[0]
    out = np.zeros((t_len, CLASSES_NUM), dtype=np.float32)

    valid = voiced & (f0_hz > 0)
    if not np.any(valid):
        return out

    m = np.full(t_len, np.nan, dtype=np.float64)
    m[valid] = 69.0 + 12.0 * np.log2(f0_hz[valid] / 440.0)

    in_range = valid & (m >= BEGIN_NOTE) & (m <= BEGIN_NOTE + CLASSES_NUM - 1)
    idx = np.where(in_range)[0]
    if idx.size == 0:
        return out

    notes = BEGIN_NOTE + np.arange(CLASSES_NUM, dtype=np.float64)  # [88]
    diff = notes[None, :] - m[idx][:, None]  # [n, 88]
    vals = np.exp(-(diff**2) / sigma)
    vals[vals < floor] = 0.0
    out[idx, :] = vals.astype(np.float32)
    return out


def voice_roll_from_npz(
    path: Path, start_frame: int, n_frames: int, sigma: float = 0.5, floor: float = 0.01
) -> np.ndarray:
    """A voice_roll window from a pitch .npz, padding file edges with unvoiced frames."""
    track = load_pitch(path)
    t_total = track.f0_hz.shape[0]

    f0 = np.zeros(n_frames, dtype=np.float32)
    voiced = np.zeros(n_frames, dtype=bool)

    src_start = max(start_frame, 0)
    src_end = min(start_frame + n_frames, t_total)
    if src_start < src_end:
        dst_start = src_start - start_frame
        dst_end = dst_start + (src_end - src_start)
        f0[dst_start:dst_end] = track.f0_hz[src_start:src_end]
        voiced[dst_start:dst_end] = track.voiced[src_start:src_end]

    return voice_roll(f0, voiced, sigma=sigma, floor=floor)
