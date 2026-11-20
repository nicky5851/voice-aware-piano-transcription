"""Protocol §9 (statistics) as tested code (spec WP2 §7).

Input is a long per-clip table with columns system, seed, clip_id,
musdb_song, condition, and the ClipScores fields. aggregate() produces the
per-clip, seed-averaged, pooled-condition table that every primary statistic
reads from. paired_bootstrap() is the one resampling engine behind all of
them, so every comparison in a decision uses the same paired draw.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VOCAL_CONDITIONS = ("-6", "0", "+6")
VOCAL_VPR_DB = {"-6": -6.0, "0": 0.0, "+6": 6.0}

RATE_COLS = ["onset_p", "onset_r", "onset_f1", "onoff_f1"]
COUNT_COLS = ["n_ref", "n_est", "n_match", "n_fp", "n_leak", "n_leak_oct"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Seeds averaged (rates) / summed (counts); adds the 'pooled' pseudo-condition (protocol §9.2)."""
    group_cols = ["system", "clip_id", "musdb_song", "condition"]
    agg_spec = {c: "mean" for c in RATE_COLS}
    agg_spec.update({c: "sum" for c in COUNT_COLS})
    agg_spec["dur_s"] = "first"  # same clip/condition -> same duration regardless of seed
    seed_avg = df.groupby(group_cols, as_index=False).agg(agg_spec)

    vocal = seed_avg[seed_avg["condition"].isin(VOCAL_CONDITIONS)]
    pooled_rate = vocal.groupby(["system", "clip_id", "musdb_song"], as_index=False)[RATE_COLS].mean()
    pooled_count = vocal.groupby(["system", "clip_id", "musdb_song"], as_index=False)[COUNT_COLS + ["dur_s"]].sum()
    pooled = pooled_rate.merge(pooled_count, on=["system", "clip_id", "musdb_song"])
    pooled["condition"] = "pooled"

    out = pd.concat([seed_avg, pooled], ignore_index=True)
    out["lpm"] = out["n_leak"] / (out["dur_s"] / 60.0)
    out["fpm"] = out["n_fp"] / (out["dur_s"] / 60.0)
    out["lr"] = np.where(out["n_fp"] > 0, out["n_leak"] / out["n_fp"], np.nan)
    return out


# --- Bootstrap engine ---------------------------------------------------


@dataclass
class BootstrapResult:
    point: float
    ci95: tuple[float, float]
    ci9875: tuple[float, float]
    estimates: np.ndarray


def paired_bootstrap(stat_fn, n_units: int, B: int = 20000, seed: int = 0, cluster_index_lists=None) -> BootstrapResult:
    """One [B, n_units] (or [B, n_clusters]) index draw, reused by stat_fn for every call.

    stat_fn(idx: np.ndarray[int]) -> float, where idx indexes the 0..n_units-1 (or, in
    cluster mode, 0..N_clips-1 after cluster expansion) per-unit arrays stat_fn closes over.
    """
    rng = np.random.default_rng(seed)
    if cluster_index_lists is None:
        point = stat_fn(np.arange(n_units))
        idx_matrix = rng.integers(0, n_units, size=(B, n_units))
        estimates = np.fromiter((stat_fn(idx_matrix[b]) for b in range(B)), dtype=float, count=B)
    else:
        m = len(cluster_index_lists)
        point = stat_fn(np.concatenate(cluster_index_lists))
        cluster_draws = rng.integers(0, m, size=(B, m))
        estimates = np.empty(B, dtype=float)
        for b in range(B):
            idx = np.concatenate([cluster_index_lists[j] for j in cluster_draws[b]])
            estimates[b] = stat_fn(idx)

    ci95 = tuple(np.percentile(estimates, [2.5, 97.5]))
    ci9875 = tuple(np.percentile(estimates, [0.625, 99.375]))
    return BootstrapResult(point=float(point), ci95=ci95, ci9875=ci9875, estimates=estimates)


def _mean_diff_fn(a: np.ndarray, b: np.ndarray):
    def stat_fn(idx):
        return float(np.mean(a[idx] - b[idx]))

    return stat_fn


def _mean_fn(a: np.ndarray):
    def stat_fn(idx):
        return float(np.mean(a[idx]))

    return stat_fn


def _ratio_of_sums_diff_fn(leak_a: np.ndarray, fp_a: np.ndarray, leak_b: np.ndarray, fp_b: np.ndarray):
    def stat_fn(idx):
        ra = leak_a[idx].sum() / fp_a[idx].sum()
        rb = leak_b[idx].sum() / fp_b[idx].sum()
        return float(ra - rb)

    return stat_fn


def _ratio_of_means_fn(a: np.ndarray, b: np.ndarray):
    def stat_fn(idx):
        return float(np.mean(a[idx]) / np.mean(b[idx]))

    return stat_fn


def ols_slope_3pt(y_m6: np.ndarray, y_0: np.ndarray, y_p6: np.ndarray, x=(-6.0, 0.0, 6.0)) -> np.ndarray:
    """Per-clip OLS slope of y on x over 3 points (protocol §9.4 H1b), general form."""
    x_arr = np.asarray(x, dtype=float)
    xbar = x_arr.mean()
    denom = float(np.sum((x_arr - xbar) ** 2))
    y = np.stack([y_m6, y_0, y_p6], axis=0)  # [3, n]
    ybar = y.mean(axis=0)
    num = np.sum((x_arr[:, None] - xbar) * (y - ybar[None, :]), axis=0)
    return num / denom


# --- Data alignment ------------------------------------------------------


def _pivot(agg: pd.DataFrame, system: str, condition: str, col: str) -> pd.Series:
    sub = agg[(agg["system"] == system) & (agg["condition"] == condition)].set_index("clip_id")
    if sub.index.has_duplicates:
        raise ValueError(f"duplicate clip_id for system={system} condition={condition}")
    return sub[col]


def aligned_arrays(agg: pd.DataFrame, specs: list[tuple[str, str, str]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """specs: [(system, condition, col), ...]; asserts every series shares the same clip_id set."""
    series = [_pivot(agg, s, c, col) for (s, c, col) in specs]
    idx = series[0].index
    for s in series[1:]:
        if set(s.index) != set(idx):
            raise ValueError("misaligned clip_id sets across systems/conditions")
    series = [s.loc[idx] for s in series]
    return idx.to_numpy(), [s.to_numpy(dtype=float) for s in series]


def cluster_index_lists_for(clip_ids: np.ndarray, musdb_song_by_clip: dict) -> list[np.ndarray]:
    songs = sorted({musdb_song_by_clip[c] for c in clip_ids})
    pos = {c: i for i, c in enumerate(clip_ids)}
    return [np.array([pos[c] for c in clip_ids if musdb_song_by_clip[c] == song]) for song in songs]


# --- Primary statistics (protocol §9.4) ---------------------------------


def compute_primary_statistics(
    agg: pd.DataFrame, pretrained: str = "P", lam0: str = "lam0", lam1: str = "lam1",
    B: int = 20000, seed: int = 0, cluster: str | None = None,
) -> dict[str, BootstrapResult]:
    cluster_lists_for_clips = None  # built lazily per statistic, since clip sets can differ by system pair
    results: dict[str, BootstrapResult] = {}

    def maybe_cluster(agg_sub: pd.DataFrame, clip_ids: np.ndarray) -> list | None:
        if cluster != "musdb_song":
            return None
        song_by_clip = dict(zip(agg_sub["clip_id"], agg_sub["musdb_song"]))
        return cluster_index_lists_for(clip_ids, song_by_clip)

    # H1a: P pooled F1 - P clean F1
    clip_ids, (f1_pooled_p, f1_clean_p) = aligned_arrays(
        agg, [(pretrained, "pooled", "onset_f1"), (pretrained, "clean", "onset_f1")]
    )
    clusters = maybe_cluster(agg[agg["system"] == pretrained], clip_ids)
    results["h1a"] = paired_bootstrap(_mean_diff_fn(f1_pooled_p, f1_clean_p), len(clip_ids), B, seed, clusters)

    # H1b: mean per-clip OLS slope of P's onset F1 on VPR
    clip_ids, (f1_m6, f1_0, f1_p6) = aligned_arrays(
        agg, [(pretrained, "-6", "onset_f1"), (pretrained, "0", "onset_f1"), (pretrained, "+6", "onset_f1")]
    )
    slopes = ols_slope_3pt(f1_m6, f1_0, f1_p6)
    clusters = maybe_cluster(agg[agg["system"] == pretrained], clip_ids)
    results["h1b"] = paired_bootstrap(_mean_fn(slopes), len(clip_ids), B, seed, clusters)

    # H2: lam0 - P, pooled F1
    clip_ids, (f1_lam0, f1_p) = aligned_arrays(agg, [(lam0, "pooled", "onset_f1"), (pretrained, "pooled", "onset_f1")])
    clusters = maybe_cluster(agg[agg["system"] == lam0], clip_ids)
    results["h2"] = paired_bootstrap(_mean_diff_fn(f1_lam0, f1_p), len(clip_ids), B, seed, clusters)

    # H3a: pooled leak-rate ratio, lam1 - lam0
    clip_ids, (leak1, fp1, leak0, fp0) = aligned_arrays(
        agg, [(lam1, "pooled", "n_leak"), (lam1, "pooled", "n_fp"), (lam0, "pooled", "n_leak"), (lam0, "pooled", "n_fp")]
    )
    clusters = maybe_cluster(agg[agg["system"] == lam1], clip_ids)
    results["h3a"] = paired_bootstrap(_ratio_of_sums_diff_fn(leak1, fp1, leak0, fp0), len(clip_ids), B, seed, clusters)

    # H3b: lam1 - lam0, pooled LPM
    clip_ids, (lpm1, lpm0) = aligned_arrays(agg, [(lam1, "pooled", "lpm"), (lam0, "pooled", "lpm")])
    clusters = maybe_cluster(agg[agg["system"] == lam1], clip_ids)
    results["h3b"] = paired_bootstrap(_mean_diff_fn(lpm1, lpm0), len(clip_ids), B, seed, clusters)

    # H3c: lam1 - lam0, pooled F1 (accuracy guard)
    clip_ids, (f1_lam1, f1_lam0b) = aligned_arrays(agg, [(lam1, "pooled", "onset_f1"), (lam0, "pooled", "onset_f1")])
    clusters = maybe_cluster(agg[agg["system"] == lam1], clip_ids)
    results["h3c"] = paired_bootstrap(_mean_diff_fn(f1_lam1, f1_lam0b), len(clip_ids), B, seed, clusters)

    # H3d: ratio of means FPM(lam1)/FPM(lam0), pooled
    clip_ids, (fpm1, fpm0) = aligned_arrays(agg, [(lam1, "pooled", "fpm"), (lam0, "pooled", "fpm")])
    clusters = maybe_cluster(agg[agg["system"] == lam1], clip_ids)
    results["h3d"] = paired_bootstrap(_ratio_of_means_fn(fpm1, fpm0), len(clip_ids), B, seed, clusters)

    # H4: lam1 - P, clean F1
    clip_ids, (f1_lam1_clean, f1_p_clean) = aligned_arrays(agg, [(lam1, "clean", "onset_f1"), (pretrained, "clean", "onset_f1")])
    clusters = maybe_cluster(agg[agg["system"] == lam1], clip_ids)
    results["h4"] = paired_bootstrap(_mean_diff_fn(f1_lam1_clean, f1_p_clean), len(clip_ids), B, seed, clusters)

    return results


# --- Decision rules (protocol §9.4, §9.5) -------------------------------

# (bound, op, threshold): "upper"/"lower" CI bound compared against threshold.
DECISION_RULES = {
    "h1a": ("upper", "<", 0.0),
    "h1b": ("upper", "<", 0.0),
    "h2": ("lower", ">", 0.0),
    "h3a": ("upper", "<", 0.0),
    "h3b": ("upper", "<", 0.0),
    "h3c": ("lower", ">", -1.0),
    "h3d": ("upper", "<", 1.05),
    "h4": ("lower", ">", -1.0),
}

HYPOTHESIS_COMPONENTS = {
    "H1": ["h1a", "h1b"],
    "H2": ["h2"],
    "H3": ["h3a", "h3b", "h3c", "h3d"],
    "H4": ["h4"],
}

# SESOI (§9.5): |point estimate| must reach this to be "practically meaningful".
# h3a/h3b are relative-change SESOIs (10% vs lam0), not absolute: p_rel below.
SESOI_ABS = {"h1a": 1.0, "h1b": 0.1}
SESOI_REL = {"h3a": -0.10, "h3b": -0.10}


def _bound_passes(ci9875: tuple[float, float], bound: str, op: str, threshold: float) -> bool:
    value = ci9875[1] if bound == "upper" else ci9875[0]
    return value < threshold if op == "<" else value > threshold


def decide(
    results: dict[str, BootstrapResult], baseline_for_rel: dict[str, float] | None = None, h3_adequately_tested: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies the frozen pass rules, intersection-union per hypothesis, and SESOI labels."""
    rows = []
    for key, (bound, op, threshold) in DECISION_RULES.items():
        r = results[key]
        passed = _bound_passes(r.ci9875, bound, op, threshold)
        sesoi_label = None
        if passed:
            if key in SESOI_ABS:
                sesoi_label = "supported, practically meaningful" if abs(r.point) >= SESOI_ABS[key] else "supported, below SESOI"
            elif key in SESOI_REL and baseline_for_rel is not None and key in baseline_for_rel and baseline_for_rel[key]:
                rel = r.point / baseline_for_rel[key]
                sesoi_label = "supported, practically meaningful" if rel <= SESOI_REL[key] else "supported, below SESOI"
            else:
                sesoi_label = "supported"
        rows.append(
            {
                "component": key,
                "estimate": r.point,
                "ci95_lo": r.ci95[0],
                "ci95_hi": r.ci95[1],
                "ci9875_lo": r.ci9875[0],
                "ci9875_hi": r.ci9875[1],
                "rule": f"{bound} bound {op} {threshold}",
                "pass": passed,
                "sesoi_label": sesoi_label,
            }
        )
    components_df = pd.DataFrame(rows)

    decision_rows = []
    for hyp, comps in HYPOTHESIS_COMPONENTS.items():
        comp_pass = components_df.set_index("component").loc[comps, "pass"]
        h_pass = bool(comp_pass.all())
        note = None
        if hyp == "H3" and not h3_adequately_tested:
            h_pass = False
            note = "not adequately tested (manipulation check failed)"
        decision_rows.append({"hypothesis": hyp, "pass": h_pass, "components": comps, "note": note})
    decisions_df = pd.DataFrame(decision_rows)

    return components_df, decisions_df
