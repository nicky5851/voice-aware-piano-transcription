"""Validation and test manifests: IDs and offsets only, no ground truth (protocol §6.4, spec WP2 §4).

Non-silence and all seeded choices here use audio only -- never MIDI -- per
the ground-truth isolation rule enforced by tests/test_isolation.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyloudnorm
import soundfile as sf

from vapt.constants import SAMPLE_RATE
from vapt.data.checksums import sha256_file
from vapt.mixing.mixer import MixInfo, SilentSegmentError, mix

CLIP_DUR_S = 30.0
N_VAL_PERFORMANCES = 75
MAX_CANDIDATES = 100
NON_SILENT_LUFS_FLOOR = -50.0
SUBWINDOW_RMS_FLOOR_DBFS = -60.0
CONDITIONS = ("clean", "-6", "0", "+6")
CONDITION_VPR_DB = {"-6": -6.0, "0": 0.0, "+6": 6.0}

# MAESTRO's split attribute spells "validation" in full; MUSDB's split naming (WP1) uses "val".
MAESTRO_SPLIT_NAMES = {"val": "validation", "test": "test"}
MUSDB_SPLIT_NAMES = {"val": "val", "test": "test"}


def list_split_performances(hdf5s_dir: Path, maestro_split: str) -> list[tuple[str, str, float]]:
    """Sorted (year, hdf5_name, duration_s) for every performance in one MAESTRO split."""
    out = []
    for year_dir in sorted(p for p in hdf5s_dir.iterdir() if p.is_dir()):
        for hdf5_path in sorted(year_dir.glob("*.h5")):
            with h5py.File(hdf5_path, "r") as hf:
                if hf.attrs["split"].decode() != maestro_split:
                    continue
                year = hf.attrs["year"].decode()
                duration = float(hf.attrs["duration"])
            out.append((year, hdf5_path.name, duration))
    return out


def _load_piano_window(hdf5s_dir: Path, year: str, hdf5_name: str, start_s: float, dur_s: float) -> np.ndarray:
    import sys

    third_party_utils = Path(__file__).resolve().parents[3] / "third_party" / "piano_transcription" / "utils"
    p = str(third_party_utils)
    if p not in sys.path:
        sys.path.insert(0, p)
    from utilities import int16_to_float32

    with h5py.File(hdf5s_dir / year / hdf5_name, "r") as hf:
        start_sample = int(round(start_s * SAMPLE_RATE))
        n_samples = int(round(dur_s * SAMPLE_RATE))
        end_sample = min(start_sample + n_samples, hf["waveform"].shape[0])
        waveform = int16_to_float32(hf["waveform"][start_sample:end_sample])
    if waveform.shape[0] < n_samples:
        waveform = np.pad(waveform, (0, n_samples - waveform.shape[0]))
    return waveform.astype(np.float32)


def _is_non_silent(audio: np.ndarray, meter: pyloudnorm.Meter) -> bool:
    lufs = meter.integrated_loudness(audio.astype(np.float64))
    if np.isinf(lufs) or lufs <= NON_SILENT_LUFS_FLOOR:
        return False
    sub_n = SAMPLE_RATE  # 1s sub-windows
    for start in range(0, audio.shape[0] - sub_n + 1, sub_n):
        sub = audio[start : start + sub_n]
        rms = float(np.sqrt(np.mean(sub.astype(np.float64) ** 2) + 1e-20))
        if 20.0 * np.log10(rms + 1e-12) < SUBWINDOW_RMS_FLOOR_DBFS:
            return False
    return True


def choose_non_silent_start(
    hdf5s_dir: Path, year: str, hdf5_name: str, duration: float, meter: pyloudnorm.Meter, seed: int, idx: int
) -> tuple[float, bool]:
    """Up to MAX_CANDIDATES seeded starts; falls back to the loudest if none is non-silent."""
    max_start = max(duration - CLIP_DUR_S, 0.0)
    rng = np.random.default_rng([seed, idx])
    candidates = rng.uniform(0.0, max_start, size=MAX_CANDIDATES) if max_start > 0 else np.zeros(MAX_CANDIDATES)

    best_start, best_lufs = None, -np.inf
    for start_s in candidates:
        audio = _load_piano_window(hdf5s_dir, year, hdf5_name, float(start_s), CLIP_DUR_S)
        if _is_non_silent(audio, meter):
            return float(start_s), False
        lufs = meter.integrated_loudness(audio.astype(np.float64))
        if not np.isinf(lufs) and lufs > best_lufs:
            best_start, best_lufs = float(start_s), lufs
    return (best_start if best_start is not None else 0.0), True


def round_robin_windows(windows_by_song: dict[str, list[dict]], seed: int, n_needed: int) -> list[dict]:
    """Balanced round-robin assignment of (song, start_s) windows (spec WP2 §4)."""
    songs_sorted = sorted(windows_by_song.keys())
    rng = np.random.default_rng(seed)
    song_order = [songs_sorted[i] for i in rng.permutation(len(songs_sorted))]

    per_song_windows: dict[str, list[dict]] = {}
    for song in songs_sorted:  # fixed order: deterministic regardless of song_order
        wins = windows_by_song[song]
        perm = rng.permutation(len(wins))
        per_song_windows[song] = [wins[i] for i in perm]

    sequence: list[dict] = []
    pointers = {s: 0 for s in song_order}
    remaining = list(song_order)
    while remaining:
        nxt = []
        for song in remaining:
            sequence.append(per_song_windows[song][pointers[song]])
            pointers[song] += 1
            if pointers[song] < len(per_song_windows[song]):
                nxt.append(song)
        remaining = nxt

    if not sequence:
        raise ValueError("no voiced windows available for round robin")
    return [dict(sequence[i % len(sequence)]) for i in range(n_needed)]


def build_manifest(
    hdf5s_dir: Path,
    voiced_windows_30s: list[dict],
    which: str,  # "val" | "test"
    seed: int = 0,
) -> list[dict]:
    maestro_split = MAESTRO_SPLIT_NAMES[which]
    performances = list_split_performances(hdf5s_dir, maestro_split)
    if which == "val":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(performances), size=min(N_VAL_PERFORMANCES, len(performances)), replace=False)
        idx = sorted(int(i) for i in idx)
        performances = [performances[i] for i in idx]

    meter = pyloudnorm.Meter(SAMPLE_RATE)
    windows_by_song: dict[str, list[dict]] = {}
    for w in voiced_windows_30s:
        windows_by_song.setdefault(w["song"], []).append(w)
    assigned_windows = round_robin_windows(windows_by_song, seed=seed, n_needed=len(performances))

    entries = []
    for i, ((year, hdf5_name, duration), window) in enumerate(zip(performances, assigned_windows)):
        start_s, fallback_flagged = choose_non_silent_start(hdf5s_dir, year, hdf5_name, duration, meter, seed, i)
        for condition in CONDITIONS:
            entries.append(
                {
                    "clip_id": f"{which}_{i:04d}_{condition}",
                    "maestro_file": hdf5_name,
                    "maestro_year": year,
                    "piano_start_s": start_s,
                    "dur_s": CLIP_DUR_S,
                    "musdb_song": window["song"],
                    "voice_start_s": window["start_s"],
                    "condition": condition,
                    "seed": seed,
                    "non_silent_fallback_flagged": fallback_flagged,
                }
            )
    return entries


def render(entry: dict, hdf5s_dir: Path, vocals16k_dir: Path, musdb_split: str) -> tuple[np.ndarray, MixInfo | None]:
    """Deterministic render of one manifest entry. musdb_split is 'val' or 'test' (MUSDB naming)."""
    piano = _load_piano_window(hdf5s_dir, entry["maestro_year"], entry["maestro_file"], entry["piano_start_s"], entry["dur_s"])
    if entry["condition"] == "clean":
        return piano, None

    meter = pyloudnorm.Meter(SAMPLE_RATE)
    flac_path = vocals16k_dir / musdb_split / f"{entry['musdb_song']}.flac"
    n_samples = int(round(entry["dur_s"] * SAMPLE_RATE))
    voice, sr = sf.read(flac_path, start=int(round(entry["voice_start_s"] * SAMPLE_RATE)), frames=n_samples)
    assert sr == SAMPLE_RATE
    if voice.shape[0] < n_samples:
        voice = np.pad(voice, (0, n_samples - voice.shape[0]))
    vpr_db = CONDITION_VPR_DB[entry["condition"]]
    audio, info = mix(piano, voice.astype(np.float32), vpr_db, meter)
    return audio, info


def write_manifest_with_checksum(entries: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2))
    digest = sha256_file(out_path)
    out_path.with_suffix(out_path.suffix + ".sha256").write_text(f"{digest}  {out_path.name}\n")
