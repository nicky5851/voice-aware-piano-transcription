"""Voice-head decoding and metrics (protocol §5, §8.1; spec WP2 §6)."""
from __future__ import annotations

import mir_eval
import numpy as np
from sklearn.metrics import roc_auc_score

from vapt.constants import BEGIN_NOTE, CLASSES_NUM
from vapt.data.pitch import PitchTrack

VOICING_THRESHOLD = 0.5
KEYS = BEGIN_NOTE + np.arange(CLASSES_NUM)  # MIDI notes 21..108


def decode_voice(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[T, 88] -> (v_t [T], f_hz [T]), protocol §5."""
    t_len = y.shape[0]
    v_t = y.max(axis=1)
    k_star = y.argmax(axis=1)

    f_hz = np.zeros(t_len, dtype=np.float64)
    lo = np.clip(k_star - 1, 0, CLASSES_NUM - 1)
    hi = np.clip(k_star + 1, 0, CLASSES_NUM - 1)
    for t in range(t_len):
        ks = np.arange(lo[t], hi[t] + 1)
        weights = y[t, ks]
        total = weights.sum()
        k_hat = float(np.dot(ks, weights) / total) if total > 0 else float(k_star[t])
        k_hat = float(np.clip(k_hat, 0, CLASSES_NUM - 1))
        f_hz[t] = 440.0 * 2.0 ** ((BEGIN_NOTE + k_hat - 69.0) / 12.0)
    return v_t, f_hz


def _ref_voicing_and_cents(ref: PitchTrack) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 69.0 + 12.0 * np.log2(np.where(ref.f0_hz > 0, ref.f0_hz, 1.0) / 440.0)
    in_range = (m >= BEGIN_NOTE) & (m <= BEGIN_NOTE + CLASSES_NUM - 1)
    ref_voicing = ref.voiced & in_range
    ref_f0_safe = np.where(ref.f0_hz > 0, ref.f0_hz, 10.0)
    ref_cent = mir_eval.melody.hz2cents(ref_f0_safe)
    return ref_voicing, ref_cent


def voice_metrics(y: np.ndarray, ref: PitchTrack) -> dict:
    v_t, f_hz = decode_voice(y)
    ref_voicing, ref_cent = _ref_voicing_and_cents(ref)

    est_voicing = v_t >= VOICING_THRESHOLD
    est_cent = mir_eval.melody.hz2cents(f_hz)

    vr, vfa = mir_eval.melody.voicing_measures(ref_voicing, est_voicing)
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    oa = mir_eval.melody.overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)

    auroc = float("nan")
    if ref_voicing.any() and not ref_voicing.all():
        auroc = float(roc_auc_score(ref_voicing.astype(int), v_t))

    return {
        "auroc": auroc,
        "voicing_recall": float(vr),
        "voicing_false_alarm": float(vfa),
        "rpa": float(rpa),
        "rca": float(rca),
        "oa": float(oa),
        "n_voiced": int(ref_voicing.sum()),
    }


def prior_baselines(train_rolls_mean: np.ndarray, ref: PitchTrack) -> dict:
    """All-zeros and constant-prior (per-key training frequency) heads, protocol §8.1."""
    t_len = ref.voiced.shape[0]
    zeros = np.zeros((t_len, CLASSES_NUM), dtype=np.float32)
    const = np.broadcast_to(train_rolls_mean.astype(np.float32), (t_len, CLASSES_NUM))
    return {
        "all_zeros": voice_metrics(zeros, ref),
        "constant_prior": voice_metrics(const, ref),
    }
