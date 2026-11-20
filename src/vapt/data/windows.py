"""The voiced-window index (spec WP1 §4.6)."""
from __future__ import annotations

import numpy as np


def voiced_windows(
    voiced: np.ndarray, win_s: float, hop_s: float = 1.0, min_frac: float = 0.5, fps: int = 100
) -> list[float]:
    """Start times (seconds) of fixed-length windows whose voiced-frame fraction is >= min_frac."""
    win_frames = int(round(win_s * fps))
    hop_frames = int(round(hop_s * fps))
    t_len = voiced.shape[0]

    starts: list[float] = []
    if win_frames <= 0:
        return starts

    t = 0
    while t + win_frames <= t_len:
        frac = float(np.mean(voiced[t : t + win_frames]))
        if frac >= min_frac:
            starts.append(t / fps)
        t += hop_frames
    return starts
