"""The seeded example stream used for training (protocol §7.4, spec WP2 §3).

Statelessness by construction: every draw for example (step, slot) comes from
a fresh `np.random.default_rng([seed, step, slot, stream_id, ...])`. batch_specs
is therefore a pure function of (seed, step); resume needs no RNG state and the
DataLoader worker count cannot change the stream. lambda is never an input here.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from vapt.constants import SAMPLE_RATE
from vapt.data.targets import voice_roll_from_npz
from vapt.mixing.mixer import SilentSegmentError, mix

SEGMENT_SECONDS = 10.0
HOP_SECONDS = 1.0  # Kong's config.hop_seconds
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * SAMPLE_RATE)
FRAMES_PER_SEGMENT = 1001  # TargetProcessor: int(round(segment_seconds * fps)) + 1

CLEAN_PROB = 0.2
VPR_LOW, VPR_HIGH = -9.0, 9.0

# Stream ids: one per kind of draw, so each is independent of the others.
PIANO_STREAM = 0
VOICE_STREAM = 1
VPR_STREAM = 2
CLEAN_STREAM = 3
REDRAW_STREAM = 4

THIRD_PARTY_UTILS = Path(__file__).resolve().parents[3] / "third_party" / "piano_transcription" / "utils"


def _ensure_vendored_on_path() -> None:
    import sys

    p = str(THIRD_PARTY_UTILS)
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass(frozen=True)
class ExampleSpec:
    step: int
    slot: int
    piano_meta: tuple  # (year, hdf5_name, start_time), Kong's MaestroDataset meta
    voice_song: str | None
    voice_start_s: float | None
    vpr_db: float | None
    clean: bool


def build_piano_segment_list(hdf5s_dir: Path, split: str) -> list[tuple[str, str, float]]:
    """Kong's Sampler segment list (data_generator.py Sampler.__init__), for one split.

    # ENGINEERING CHOICE: Kong's Sampler walks the directory with os.walk, whose
    # order is filesystem-dependent and not guaranteed stable across machines or
    # re-extractions. Reproducibility across pods matters more than matching his
    # exact (arbitrary) walk order, so files are visited in sorted (year, name)
    # order instead. The segment list's *contents* are identical either way; only
    # the list's index order differs, and that order is itself randomly permuted
    # by the stream (never relied on directly), so this has no scientific effect.
    """
    segment_list: list[tuple[str, str, float]] = []
    for year_dir in sorted(p for p in hdf5s_dir.iterdir() if p.is_dir()):
        for hdf5_path in sorted(year_dir.glob("*.h5")):
            with h5py.File(hdf5_path, "r") as hf:
                if hf.attrs["split"].decode() != split:
                    continue
                year = hf.attrs["year"].decode()
                duration = float(hf.attrs["duration"])
            start_time = 0.0
            while start_time + SEGMENT_SECONDS < duration:
                segment_list.append((year, hdf5_path.name, start_time))
                start_time += HOP_SECONDS
    return segment_list


class ExampleStream:
    """Pure function (seed, step) -> batch of ExampleSpec. See module docstring."""

    def __init__(self, seed: int, piano_segments: Sequence[tuple[str, str, float]], voice_windows: Sequence[dict], batch_size: int = 12):
        self.seed = seed
        self.piano_segments = list(piano_segments)
        self.voice_windows = list(voice_windows)
        self.batch_size = batch_size
        if not self.piano_segments:
            raise ValueError("piano_segments must be non-empty")
        if not self.voice_windows:
            raise ValueError("voice_windows must be non-empty")

    def _piano_for_global_index(self, g: int) -> tuple[str, str, float]:
        n = len(self.piano_segments)
        epoch, pos = divmod(g, n)
        perm = np.random.default_rng([self.seed, epoch, PIANO_STREAM]).permutation(n)
        return self.piano_segments[int(perm[pos])]

    def spec_for(self, step: int, slot: int) -> ExampleSpec:
        g = step * self.batch_size + slot
        piano_meta = self._piano_for_global_index(g)

        clean = bool(np.random.default_rng([self.seed, step, slot, CLEAN_STREAM]).random() < CLEAN_PROB)

        if clean:
            voice_song = None
            voice_start_s = None
            vpr_db = None
        else:
            voice_idx = int(
                np.random.default_rng([self.seed, step, slot, VOICE_STREAM]).integers(0, len(self.voice_windows))
            )
            window = self.voice_windows[voice_idx]
            voice_song = str(window["song"])
            voice_start_s = float(window["start_s"])
            vpr_db = float(np.random.default_rng([self.seed, step, slot, VPR_STREAM]).uniform(VPR_LOW, VPR_HIGH))

        return ExampleSpec(
            step=step,
            slot=slot,
            piano_meta=piano_meta,
            voice_song=voice_song,
            voice_start_s=voice_start_s,
            vpr_db=vpr_db,
            clean=clean,
        )

    def batch_specs(self, step: int) -> list[ExampleSpec]:
        return [self.spec_for(step, slot) for slot in range(self.batch_size)]

    def redraw_voice(self, step: int, slot: int, redraw_count: int) -> tuple[str, float]:
        """The next voice-window draw from the same (step, slot)'s stream, for a SilentSegmentError redraw."""
        voice_idx = int(
            np.random.default_rng([self.seed, step, slot, REDRAW_STREAM, redraw_count]).integers(
                0, len(self.voice_windows)
            )
        )
        window = self.voice_windows[voice_idx]
        return str(window["song"]), float(window["start_s"])


class MixtureDataset(Dataset):
    """Loads a piano segment (Kong's unchanged target processor) and a voice window, mixes them."""

    def __init__(self, hdf5s_dir: Path, vocals16k_dir: Path, pitch_dir: Path, stream: ExampleStream, meter, max_redraws: int = 8):
        _ensure_vendored_on_path()
        from utilities import TargetProcessor, int16_to_float32

        self.hdf5s_dir = Path(hdf5s_dir)
        self.vocals16k_dir = Path(vocals16k_dir)
        self.pitch_dir = Path(pitch_dir)
        self.stream = stream
        self.meter = meter
        self.max_redraws = max_redraws
        self._int16_to_float32 = int16_to_float32
        self.target_processor = TargetProcessor(SEGMENT_SECONDS, 100, 21, 88)

    def _load_piano(self, piano_meta: tuple[str, str, float]) -> tuple[np.ndarray, dict]:
        year, hdf5_name, start_time = piano_meta
        hdf5_path = self.hdf5s_dir / year / hdf5_name
        with h5py.File(hdf5_path, "r") as hf:
            start_sample = int(start_time * SAMPLE_RATE)
            end_sample = start_sample + SEGMENT_SAMPLES
            if end_sample > hf["waveform"].shape[0]:
                start_sample = max(hf["waveform"].shape[0] - SEGMENT_SAMPLES, 0)
                end_sample = start_sample + SEGMENT_SAMPLES
            waveform = self._int16_to_float32(hf["waveform"][start_sample:end_sample])
            midi_events = [e.decode() for e in hf["midi_event"][:]]
            midi_events_time = hf["midi_event_time"][:]
            target_dict, _, _ = self.target_processor.process(start_time, midi_events_time, midi_events, extend_pedal=True)
        return waveform.astype(np.float32), target_dict

    def _load_voice_window(self, song: str, start_s: float) -> tuple[np.ndarray, np.ndarray]:
        flac_path = self.vocals16k_dir / "train" / f"{song}.flac"
        audio, sr = sf.read(flac_path, start=int(round(start_s * SAMPLE_RATE)), frames=SEGMENT_SAMPLES)
        assert sr == SAMPLE_RATE
        if audio.shape[0] < SEGMENT_SAMPLES:
            audio = np.pad(audio, (0, SEGMENT_SAMPLES - audio.shape[0]))
        start_frame = int(round(start_s * 100))
        roll = voice_roll_from_npz(self.pitch_dir / "train" / f"{song}.npz", start_frame, FRAMES_PER_SEGMENT)
        return audio.astype(np.float32), roll

    def __getitem__(self, spec: ExampleSpec) -> dict:
        waveform, target_dict = self._load_piano(spec.piano_meta)

        if spec.clean:
            voice_roll_arr = np.zeros((FRAMES_PER_SEGMENT, 88), dtype=np.float32)
            out = {"waveform": waveform, "voice_roll": voice_roll_arr, "vpr_db": None, "clean": True, "redraws": 0}
            out.update(target_dict)
            return out

        song, start_s = spec.voice_song, spec.voice_start_s
        redraws = 0
        while True:
            voice_audio, voice_roll_arr = self._load_voice_window(song, start_s)
            try:
                mixed, _info = mix(waveform, voice_audio, spec.vpr_db, self.meter)
                break
            except SilentSegmentError:
                redraws += 1
                if redraws > self.max_redraws:
                    raise
                song, start_s = self.stream.redraw_voice(spec.step, spec.slot, redraws)

        out = {"waveform": mixed, "voice_roll": voice_roll_arr, "vpr_db": spec.vpr_db, "clean": False, "redraws": redraws}
        out.update(target_dict)
        return out


class BatchSampler:
    """Yields stream.batch_specs(step) for step = start_step, start_step+1, ... ."""

    def __init__(self, stream: ExampleStream, start_step: int = 0):
        self.stream = stream
        self.start_step = start_step

    def __iter__(self):
        step = self.start_step
        while True:
            yield self.stream.batch_specs(step)
            step += 1
