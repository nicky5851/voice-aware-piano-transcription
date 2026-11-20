import numpy as np
import pandas as pd
import pytest

from vapt.eval.stats import (
    BootstrapResult,
    aggregate,
    cluster_index_lists_for,
    compute_primary_statistics,
    decide,
    ols_slope_3pt,
    paired_bootstrap,
)

CONDITIONS = ("clean", "-6", "0", "+6")
SEEDS = (0, 1, 2)


def _synthetic_df(n_clips: int, f1_means: dict, leak_prob: dict, rng: np.random.Generator, noise_sd: float = 1.5) -> pd.DataFrame:
    rows = []
    for ci in range(n_clips):
        clip_id = f"clip{ci:03d}"
        song = f"song{ci % 12}"
        clip_bias = rng.normal(0, 1.0)
        for system in ("P", "lam0", "lam1"):
            seeds = (0,) if system == "P" else SEEDS
            for sd in seeds:
                for cond in CONDITIONS:
                    f1 = float(np.clip(f1_means[system][cond] + clip_bias + rng.normal(0, noise_sd), 0, 100))
                    n_ref, n_est = 20, 19
                    n_match = min(n_est, int(round(n_est * (f1 / 100.0))))
                    n_fp = n_est - n_match
                    n_leak = 0
                    if cond != "clean" and n_fp > 0:
                        n_leak = int(rng.binomial(n_fp, leak_prob[system]))
                    rows.append(
                        {
                            "system": system, "seed": sd, "clip_id": clip_id, "musdb_song": song, "condition": cond,
                            "dur_s": 30.0, "n_ref": n_ref, "n_est": n_est, "n_match": n_match, "n_fp": n_fp,
                            "n_leak": n_leak, "n_leak_oct": n_leak,
                            "onset_p": f1, "onset_r": f1, "onset_f1": f1, "onoff_f1": f1,
                        }
                    )
    return pd.DataFrame(rows)


_F1_WITH_EFFECT = {
    "P": {"clean": 90.0, "-6": 68.0, "0": 73.0, "+6": 63.0},
    "lam0": {"clean": 88.0, "-6": 82.0, "0": 83.0, "+6": 81.0},
    "lam1": {"clean": 88.3, "-6": 82.3, "0": 83.3, "+6": 81.3},
}


@pytest.fixture(scope="module")
def true_effect_agg():
    rng = np.random.default_rng(0)
    df = _synthetic_df(120, _F1_WITH_EFFECT, {"P": 0.5, "lam0": 0.30, "lam1": 0.24}, rng)
    return aggregate(df)


def test_true_h3_effect_passes(true_effect_agg):
    results = compute_primary_statistics(true_effect_agg, B=4000, seed=0)
    components, decisions = decide(results, baseline_for_rel={"h3a": -0.15, "h3b": -2.0})
    row = components.set_index("component")
    assert row.loc["h3a", "pass"]
    assert row.loc["h3b", "pass"]
    assert row.loc["h3c", "pass"]  # f1 guard: lam1 and lam0 nearly identical
    assert row.loc["h3d", "pass"]  # fpm ratio guard
    h3_decision = decisions.set_index("hypothesis").loc["H3", "pass"]
    assert h3_decision


def test_null_case_mostly_fails_h3a():
    f1_null = {
        "P": {"clean": 90.0, "-6": 68.0, "0": 73.0, "+6": 63.0},
        "lam0": {"clean": 88.0, "-6": 82.0, "0": 83.0, "+6": 81.0},
        "lam1": {"clean": 88.0, "-6": 82.0, "0": 83.0, "+6": 81.0},
    }
    n_pass = 0
    n_trials = 8
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        df = _synthetic_df(60, f1_null, {"P": 0.5, "lam0": 0.30, "lam1": 0.30}, rng)
        agg = aggregate(df)
        results = compute_primary_statistics(agg, B=2000, seed=trial)
        components, _ = decide(results)
        if components.set_index("component").loc["h3a", "pass"]:
            n_pass += 1
    assert n_pass <= 2, f"null case passed H3a in {n_pass}/{n_trials} trials, expected it to mostly fail"


def test_h1b_slope_equals_simple_form_for_symmetric_vpr():
    rng = np.random.default_rng(42)
    y_m6 = rng.normal(70, 5, size=50)
    y_0 = rng.normal(75, 5, size=50)
    y_p6 = rng.normal(65, 5, size=50)
    slopes = ols_slope_3pt(y_m6, y_0, y_p6)
    simple = (y_p6 - y_m6) / 12.0
    assert np.allclose(slopes, simple)


def test_h3d_ratio_statistic():
    a = np.array([10.0, 20.0, 30.0, 40.0])
    b = np.array([10.0, 10.0, 10.0, 10.0])
    results = {
        "h3d": paired_bootstrap(lambda idx: float(np.mean(a[idx]) / np.mean(b[idx])), 4, B=1000, seed=0)
    }
    assert abs(results["h3d"].point - 2.5) < 1e-9


def test_intersection_union_logic():
    def fake(point, lo, hi):
        return BootstrapResult(point=point, ci95=(lo, hi), ci9875=(lo, hi), estimates=np.array([point]))

    results = {
        "h1a": fake(-2.0, -3.0, -1.0),
        "h1b": fake(-0.2, -0.3, -0.05),
        "h2": fake(5.0, 2.0, 8.0),
        "h3a": fake(-0.15, -0.25, -0.05),
        "h3b": fake(-3.0, -5.0, -1.0),
        "h3c": fake(-0.3, -0.8, 0.2),  # passes the -1.0 margin
        "h3d": fake(1.01, 0.95, 1.03),
        "h4": fake(-0.5, -0.9, 0.1),
    }
    components, decisions = decide(results)
    assert decisions.set_index("hypothesis").loc["H1", "pass"]
    assert decisions.set_index("hypothesis").loc["H3", "pass"]

    # Break just one H3 component (h3d fails its <1.05 guard) -> H3 as a whole must fail.
    results["h3d"] = fake(1.10, 1.06, 1.15)
    components, decisions = decide(results)
    assert not decisions.set_index("hypothesis").loc["H3", "pass"]


def test_h3_not_adequately_tested_overrides_pass():
    def fake(point, lo, hi):
        return BootstrapResult(point=point, ci95=(lo, hi), ci9875=(lo, hi), estimates=np.array([point]))

    results = {
        "h1a": fake(-2.0, -3.0, -1.0), "h1b": fake(-0.2, -0.3, -0.05), "h2": fake(5.0, 2.0, 8.0),
        "h3a": fake(-0.15, -0.25, -0.05), "h3b": fake(-3.0, -5.0, -1.0), "h3c": fake(-0.3, -0.8, 0.2),
        "h3d": fake(1.01, 0.95, 1.03), "h4": fake(-0.5, -0.9, 0.1),
    }
    _, decisions = decide(results, h3_adequately_tested=False)
    row = decisions.set_index("hypothesis").loc["H3"]
    assert not row["pass"]
    assert row["note"] is not None


def test_sesoi_labels():
    def fake(point, lo, hi):
        return BootstrapResult(point=point, ci95=(lo, hi), ci9875=(lo, hi), estimates=np.array([point]))

    results = {
        "h1a": fake(-2.0, -3.0, -1.0),  # |−2| >= SESOI 1.0 -> practically meaningful
        "h1b": fake(-0.05, -0.08, -0.01),  # |−0.05| < SESOI 0.1 -> below SESOI
        "h2": fake(5.0, 2.0, 8.0), "h3a": fake(-0.15, -0.25, -0.05), "h3b": fake(-3.0, -5.0, -1.0),
        "h3c": fake(-0.3, -0.8, 0.2), "h3d": fake(1.01, 0.95, 1.03), "h4": fake(-0.5, -0.9, 0.1),
    }
    components, _ = decide(results)
    row = components.set_index("component")
    assert row.loc["h1a", "sesoi_label"] == "supported, practically meaningful"
    assert row.loc["h1b", "sesoi_label"] == "supported, below SESOI"


def test_bootstrap_reproducible_with_seed():
    a = np.arange(30, dtype=float)
    r1 = paired_bootstrap(lambda idx: float(np.mean(a[idx])), 30, B=500, seed=0)
    r2 = paired_bootstrap(lambda idx: float(np.mean(a[idx])), 30, B=500, seed=0)
    assert np.array_equal(r1.estimates, r2.estimates)
    assert r1.point == r2.point


def test_cluster_bootstrap_resamples_whole_songs():
    clip_ids = np.array([f"c{i}" for i in range(12)])
    song_by_clip = {clip_ids[i]: f"song{i % 3}" for i in range(12)}
    lists = cluster_index_lists_for(clip_ids, song_by_clip)
    assert len(lists) == 3
    covered = sorted(np.concatenate(lists).tolist())
    assert covered == list(range(12))
    for lst in lists:
        songs_in_cluster = {song_by_clip[clip_ids[i]] for i in lst}
        assert len(songs_in_cluster) == 1


@pytest.mark.slow
def test_ci_coverage_over_null_datasets():
    rng_master = np.random.default_rng(7)
    f1_null = {
        "P": {"clean": 90.0, "-6": 68.0, "0": 73.0, "+6": 63.0},
        "lam0": {"clean": 88.0, "-6": 82.0, "0": 83.0, "+6": 81.0},
        "lam1": {"clean": 88.0, "-6": 82.0, "0": 83.0, "+6": 81.0},
    }
    covers = 0
    n_trials = 300
    for trial in range(n_trials):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        df = _synthetic_df(40, f1_null, {"P": 0.5, "lam0": 0.30, "lam1": 0.30}, rng)
        agg = aggregate(df)
        from vapt.eval.stats import aligned_arrays, _mean_diff_fn

        clip_ids, (f1_1, f1_0) = aligned_arrays(agg, [("lam1", "pooled", "onset_f1"), ("lam0", "pooled", "onset_f1")])
        r = paired_bootstrap(_mean_diff_fn(f1_1, f1_0), len(clip_ids), B=2000, seed=trial)
        if r.ci95[0] <= 0.0 <= r.ci95[1]:
            covers += 1
    coverage = covers / n_trials
    assert 0.90 <= coverage <= 1.0
