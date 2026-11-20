"""Transcription and leakage metrics (protocol §5, §9.4; spec WP2 §5)."""
from __future__ import annotations

from dataclasses import dataclass

import mir_eval
import numpy as np

from vapt.data.pitch import PitchTrack

LEAK_WINDOW_FRAMES = 2
FPS = 100


@dataclass
class ClipScores:
    n_ref: int
    n_est: int
    n_match: int
    n_fp: int
    n_leak: int
    n_leak_oct: int
    dur_s: float
    onset_p: float
    onset_r: float
    onset_f1: float
    onoff_f1: float


def _intervals_pitches_hz(notes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros((0, 2)), np.zeros((0,))
    intervals = np.array([[n["onset_time"], n["offset_time"]] for n in notes], dtype=float)
    midi_notes = np.array([float(n["midi_note"]) for n in notes], dtype=float)
    pitches_hz = mir_eval.util.midi_to_hz(midi_notes)
    return intervals, pitches_hz


def _is_leak(voiced: np.ndarray, m_round: np.ndarray, onset_s: float, midi_pitch: int, octave: bool) -> bool:
    f = int(round(onset_s * FPS))
    lo = max(f - LEAK_WINDOW_FRAMES, 0)
    hi = min(f + LEAK_WINDOW_FRAMES, voiced.shape[0] - 1)
    if lo > hi:
        return False
    for t in range(lo, hi + 1):
        if not voiced[t]:
            continue
        if octave:
            if (m_round[t] - midi_pitch) % 12 == 0:
                return True
        else:
            if m_round[t] == midi_pitch:
                return True
    return False


def score_clip(
    ref_notes: list[dict], est_notes: list[dict], voice_track_window: PitchTrack | None, dur_s: float
) -> ClipScores:
    n_ref, n_est = len(ref_notes), len(est_notes)
    ref_i, ref_p = _intervals_pitches_hz(ref_notes)
    est_i, est_p = _intervals_pitches_hz(est_notes)

    if n_ref == 0 and n_est == 0:
        return ClipScores(
            n_ref=0, n_est=0, n_match=0, n_fp=0, n_leak=0, n_leak_oct=0, dur_s=dur_s,
            onset_p=float("nan"), onset_r=float("nan"), onset_f1=float("nan"), onoff_f1=float("nan"),
        )

    onset_p, onset_r, onset_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, offset_ratio=None
    )
    _, _, onoff_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_i, ref_p, est_i, est_p)
    matches = mir_eval.transcription.match_notes(ref_i, ref_p, est_i, est_p, offset_ratio=None)
    matched_est = {e for (_, e) in matches}
    n_match = len(matches)
    n_fp = n_est - n_match

    n_leak = 0
    n_leak_oct = 0
    if voice_track_window is not None and n_fp > 0:
        voiced = voice_track_window.voiced
        f0 = voice_track_window.f0_hz
        with np.errstate(divide="ignore", invalid="ignore"):
            m = 69.0 + 12.0 * np.log2(np.where(f0 > 0, f0, 1.0) / 440.0)
        m_round = np.round(m).astype(int)
        for idx, note in enumerate(est_notes):
            if idx in matched_est:
                continue
            midi_pitch = int(note["midi_note"])
            onset_s = float(note["onset_time"])
            if _is_leak(voiced, m_round, onset_s, midi_pitch, octave=False):
                n_leak += 1
            if _is_leak(voiced, m_round, onset_s, midi_pitch, octave=True):
                n_leak_oct += 1

    return ClipScores(
        n_ref=n_ref, n_est=n_est, n_match=n_match, n_fp=n_fp, n_leak=n_leak, n_leak_oct=n_leak_oct, dur_s=dur_s,
        onset_p=float(onset_p) * 100.0, onset_r=float(onset_r) * 100.0, onset_f1=float(onset_f1) * 100.0,
        onoff_f1=float(onoff_f1) * 100.0,
    )
