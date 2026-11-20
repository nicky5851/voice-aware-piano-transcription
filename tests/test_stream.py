import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset

from vapt.mixing.stream import ExampleStream

PIANO_SEGMENTS = [("2004", f"file{i}.h5", float(i)) for i in range(37)]
VOICE_WINDOWS = [{"song": f"song{i}", "start_s": float(i)} for i in range(13)]


def make_stream(seed: int) -> ExampleStream:
    return ExampleStream(seed, PIANO_SEGMENTS, VOICE_WINDOWS, batch_size=12)


def test_batch_specs_pure_function_of_seed_and_step():
    s1 = make_stream(0)
    s2 = make_stream(0)
    assert s1.batch_specs(5) == s2.batch_specs(5)
    assert s1.batch_specs(0) == s2.batch_specs(0)


def test_different_seeds_diverge():
    s0 = make_stream(0)
    s1 = make_stream(1)
    assert s0.batch_specs(5) != s1.batch_specs(5)


def test_resume_at_arbitrary_step_matches_fresh_instance():
    stream_full = make_stream(0)
    specs_at_137_full = stream_full.batch_specs(137)

    # Simulated resume: a brand-new instance, asked directly for step 137,
    # with no knowledge of steps 0..136.
    stream_resumed = make_stream(0)
    specs_at_137_resumed = stream_resumed.batch_specs(137)
    assert specs_at_137_full == specs_at_137_resumed


class _StepDataset(Dataset):
    def __init__(self, stream: ExampleStream, steps: list[int]):
        self.stream = stream
        self.steps = steps

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        return self.stream.batch_specs(self.steps[idx])


@pytest.mark.parametrize("num_workers", [0, 4])
def test_dataloader_worker_count_does_not_change_stream(num_workers):
    steps = list(range(20))
    ds = _StepDataset(make_stream(0), steps)
    loader = DataLoader(ds, batch_size=None, num_workers=num_workers, shuffle=False)
    got = list(loader)
    expected = [make_stream(0).batch_specs(s) for s in steps]
    assert got == expected


def test_lambda_is_not_a_constructor_or_method_argument():
    import inspect

    sig = inspect.signature(ExampleStream.__init__)
    assert "lam" not in sig.parameters and "lambda_" not in sig.parameters
    sig2 = inspect.signature(ExampleStream.batch_specs)
    assert "lam" not in sig2.parameters and "lambda_" not in sig2.parameters


def test_clean_fraction_near_target():
    stream = make_stream(0)
    n = 10000
    clean_count = sum(1 for step in range(n // 12 + 1) for spec in stream.batch_specs(step)[: n - step * 12] if spec.clean)
    frac = clean_count / n
    assert abs(frac - 0.2) <= 0.01


def test_vpr_within_range_and_none_when_clean():
    stream = make_stream(0)
    vprs = []
    for step in range(500):
        for spec in stream.batch_specs(step):
            if spec.clean:
                assert spec.vpr_db is None
                assert spec.voice_song is None
                assert spec.voice_start_s is None
            else:
                assert spec.vpr_db is not None
                vprs.append(spec.vpr_db)
    assert min(vprs) >= -9.0
    assert max(vprs) <= 9.0


def test_voice_windows_come_only_from_supplied_pool():
    stream = make_stream(0)
    songs = {w["song"] for w in VOICE_WINDOWS}
    for step in range(200):
        for spec in stream.batch_specs(step):
            if not spec.clean:
                assert spec.voice_song in songs


def test_piano_segments_without_replacement_within_epoch():
    n = len(PIANO_SEGMENTS)
    stream = make_stream(0)
    steps_per_epoch = -(-n // 12)
    seen = set()
    for step in range(steps_per_epoch):
        for spec in stream.batch_specs(step):
            g = step * 12 + spec.slot
            if g < n:
                seen.add(spec.piano_meta)
    assert len(seen) == len(set(PIANO_SEGMENTS))
