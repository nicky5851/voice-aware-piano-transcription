"""MUSDB18-HQ vocal-stem extraction and the train/val/test split (spec WP1 §4.2-4.3)."""
from __future__ import annotations

import numpy as np
import soxr

SOURCE_SR = 44100
TARGET_SR = 16000
N_TRAIN_OFFICIAL = 100
N_TEST_OFFICIAL = 50
N_VAL = 10


def resample_to_16k(audio: np.ndarray, source_sr: int = SOURCE_SR, target_sr: int = TARGET_SR) -> np.ndarray:
    """Mono 44.1kHz -> 16kHz, matching spec §4.2 step 3 exactly (soxr HQ)."""
    return soxr.resample(audio, source_sr, target_sr, quality="HQ")


def to_mono(stereo: np.ndarray) -> np.ndarray:
    """[2, T] or [T, 2] -> [T], channel-averaged (spec §4.2 step 2)."""
    if stereo.ndim == 1:
        return stereo
    if stereo.shape[0] == 2:
        return stereo.mean(axis=0)
    return stereo.mean(axis=1)


def musdb_split(train_songs: list[str], test_songs: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Official 100/50 split, with 10 train songs held out for val (spec §4.3). Deterministic."""
    train_sorted = sorted(train_songs)
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(len(train_sorted), N_VAL, replace=False)
    val_set = {train_sorted[i] for i in val_idx}
    train_final = [s for s in train_sorted if s not in val_set]
    val_final = sorted(val_set)
    return {"train": train_final, "val": val_final, "test": sorted(test_songs)}
