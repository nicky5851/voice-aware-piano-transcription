import numpy as np

from vapt.data.musdb import musdb_split, resample_to_16k


def test_resample_length():
    n = 44100 * 3  # 3s
    audio = np.zeros(n, dtype=np.float32)
    out = resample_to_16k(audio)
    expected = round(n * 16000 / 44100)
    assert abs(len(out) - expected) <= 1


def test_split_sizes_and_no_overlap():
    train = [f"train_{i}" for i in range(100)]
    test = [f"test_{i}" for i in range(50)]
    split = musdb_split(train, test)
    assert len(split["train"]) == 90
    assert len(split["val"]) == 10
    assert len(split["test"]) == 50
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"] + split["val"]).isdisjoint(split["test"])


def test_split_deterministic_across_runs():
    train = [f"train_{i}" for i in range(100)]
    test = [f"test_{i}" for i in range(50)]
    a = musdb_split(train, test)
    b = musdb_split(train, test)
    assert a == b
