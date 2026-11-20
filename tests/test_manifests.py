import json
from pathlib import Path

import numpy as np
import pytest

from vapt.data.checksums import sha256_file
from vapt.mixing.manifests import round_robin_windows, write_manifest_with_checksum

WORKSPACE = Path("/workspace/vapt/work")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_windows(counts: dict[str, int]) -> dict[str, list[dict]]:
    return {song: [{"song": song, "start_s": float(i)} for i in range(n)] for song, n in counts.items()}


def test_round_robin_is_deterministic():
    windows = _synthetic_windows({"a": 3, "b": 3, "c": 3})
    r1 = round_robin_windows(windows, seed=0, n_needed=9)
    r2 = round_robin_windows(windows, seed=0, n_needed=9)
    assert r1 == r2


def test_round_robin_different_seed_differs():
    windows = _synthetic_windows({"a": 5, "b": 5, "c": 5})
    r0 = round_robin_windows(windows, seed=0, n_needed=15)
    r1 = round_robin_windows(windows, seed=1, n_needed=15)
    assert r0 != r1


def test_round_robin_no_reuse_within_one_full_pass():
    windows = _synthetic_windows({"a": 4, "b": 4, "c": 4})
    n_total = 12
    out = round_robin_windows(windows, seed=0, n_needed=n_total)
    keys = [(w["song"], w["start_s"]) for w in out]
    assert len(set(keys)) == n_total  # every window used exactly once in one full pass


def test_round_robin_balanced_counts_differ_by_at_most_one_round():
    windows = _synthetic_windows({"a": 7, "b": 3, "c": 5})
    out = round_robin_windows(windows, seed=0, n_needed=15)
    per_song_count = {}
    for w in out:
        per_song_count[w["song"]] = per_song_count.get(w["song"], 0) + 1
    # a has 7, b has 3, c has 5 -> in one full pass of all 15, counts == supply exactly.
    assert per_song_count == {"a": 7, "b": 3, "c": 5}


def test_round_robin_wraps_without_reusing_before_exhaustion():
    windows = _synthetic_windows({"a": 2, "b": 2})
    out = round_robin_windows(windows, seed=0, n_needed=10)  # 2.5x the pool
    first_pass = out[:4]
    assert len(set((w["song"], w["start_s"]) for w in first_pass)) == 4
    assert out[4:8] == first_pass  # second pass repeats the same balanced sequence


def test_manifest_checksum_is_regenerated_identically(tmp_path):
    entries = [{"a": 1}, {"b": 2}]
    out1 = tmp_path / "m1" / "x.json"
    out2 = tmp_path / "m2" / "x.json"
    write_manifest_with_checksum(entries, out1)
    write_manifest_with_checksum(entries, out2)
    d1 = sha256_file(out1)
    d2 = sha256_file(out2)
    assert d1 == d2


_HAS_REAL_DATA = (WORKSPACE / "hdf5s" / "maestro").exists() and (WORKSPACE / "pitch" / "val" / "voiced_windows_30s.json").exists()


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="requires WP1 data on the pod's network volume")
def test_val_manifest_builds_and_vpr_within_tolerance():
    from vapt.mixing.manifests import build_manifest, render

    hdf5s_dir = WORKSPACE / "hdf5s" / "maestro"
    windows = json.loads((WORKSPACE / "pitch" / "val" / "voiced_windows_30s.json").read_text())
    entries = build_manifest(hdf5s_dir, windows, which="val", seed=0)

    entries_again = build_manifest(hdf5s_dir, windows, which="val", seed=0)
    assert entries == entries_again

    assert all(e["musdb_song"] for e in entries)
    conditions = {e["condition"] for e in entries}
    assert conditions == {"clean", "-6", "0", "+6"}

    max_err = 0.0
    errors = []
    for e in entries:
        if e["condition"] == "clean":
            continue
        _audio, info = render(e, hdf5s_dir, WORKSPACE / "vocals16k", musdb_split="val")
        err = abs(info.vpr_measured - info.vpr_target)
        errors.append({"clip_id": e["clip_id"], "err_db": err})
        max_err = max(max_err, err)

    report_path = REPO_ROOT / "results" / "data" / "wp2_val_manifest_vpr_check.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"max_err_db": max_err, "n_items": len(errors), "errors": errors}, indent=2))

    assert max_err <= 0.5, f"max VPR error {max_err:.3f} dB exceeds the 0.5 dB tolerance"
