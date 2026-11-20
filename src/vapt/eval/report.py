"""Tables and figures from the stats library's outputs (spec WP2 §7.6)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def table_t1(agg: pd.DataFrame, systems: list[str], condition: str = "pooled") -> pd.DataFrame:
    """T1: one row per system, mean onset F1 (and onset-offset F1) at the given condition."""
    sub = agg[agg["condition"] == condition]
    rows = []
    for system in systems:
        s = sub[sub["system"] == system]
        rows.append(
            {
                "system": system,
                "n_clips": len(s),
                "onset_f1_mean": s["onset_f1"].mean(),
                "onoff_f1_mean": s["onoff_f1"].mean(),
                "lpm_mean": s["lpm"].mean(),
                "fpm_mean": s["fpm"].mean(),
            }
        )
    return pd.DataFrame(rows)


def table_t2(components_df: pd.DataFrame, decisions_df: pd.DataFrame) -> pd.DataFrame:
    """T2: the primary decision table, one row per hypothesis component."""
    out = components_df.copy()
    hyp_by_component = {}
    for _, row in decisions_df.iterrows():
        for comp in row["components"]:
            hyp_by_component[comp] = (row["hypothesis"], row["pass"], row["note"])
    out["hypothesis"] = out["component"].map(lambda c: hyp_by_component[c][0])
    out["hypothesis_pass"] = out["component"].map(lambda c: hyp_by_component[c][1])
    out["hypothesis_note"] = out["component"].map(lambda c: hyp_by_component[c][2])
    return out


def table_t3(agg: pd.DataFrame, systems: list[str], seeds: list[int], df_with_seed: pd.DataFrame, condition: str = "pooled") -> pd.DataFrame:
    """T3: per-seed onset F1, from the pre-aggregation (seed-resolved) table.

    "pooled" is computed here (mean over the three vocal conditions, per
    clip and seed) since the raw per-seed table has no pooled rows yet --
    those are only added by aggregate()'s seed-averaging step.
    """
    if condition == "pooled":
        sub = df_with_seed[df_with_seed["condition"].isin(("-6", "0", "+6"))]
        sub = sub.groupby(["system", "seed", "clip_id"], as_index=False)["onset_f1"].mean()
    else:
        sub = df_with_seed[df_with_seed["condition"] == condition]
    rows = []
    for system in systems:
        for seed in seeds:
            s = sub[(sub["system"] == system) & (sub["seed"] == seed)]
            if len(s) == 0:
                continue
            rows.append({"system": system, "seed": seed, "onset_f1_mean": s["onset_f1"].mean(), "n_clips": len(s)})
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    tex_path = csv_path.with_suffix(".tex")
    tex_path.write_text(df.to_latex(index=False, float_format="%.2f"))


def figure_f1_onset_vs_vpr(agg: pd.DataFrame, systems: list[str], out_path: Path) -> None:
    """F1: onset F1 vs VPR, one line per system, clean as a separate point, 95% CI error bars (descriptive)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    vpr_order = [-6.0, 0.0, 6.0]
    vpr_cond = {-6.0: "-6", 0.0: "0", 6.0: "+6"}
    for system in systems:
        sub = agg[agg["system"] == system]
        means, sems = [], []
        for v in vpr_order:
            s = sub[sub["condition"] == vpr_cond[v]]["onset_f1"]
            means.append(s.mean())
            sems.append(s.sem() if len(s) > 1 else 0.0)
        ax.errorbar(vpr_order, means, yerr=[1.96 * s for s in sems], marker="o", label=system)
        clean = sub[sub["condition"] == "clean"]["onset_f1"]
        if len(clean) > 0:
            ax.scatter([-9.0], [clean.mean()], marker="s")
    ax.set_xlabel("VPR (dB)")
    ax.set_ylabel("Onset F1 (points)")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_f2_leakage_vs_vpr(agg: pd.DataFrame, systems: list[str], out_path: Path) -> None:
    """F2: LPM and LR vs VPR, two panels."""
    fig, (ax_lpm, ax_lr) = plt.subplots(1, 2, figsize=(10, 4))
    vpr_order = [-6.0, 0.0, 6.0]
    vpr_cond = {-6.0: "-6", 0.0: "0", 6.0: "+6"}
    for system in systems:
        sub = agg[agg["system"] == system]
        lpm_means, lr_means = [], []
        for v in vpr_order:
            s = sub[sub["condition"] == vpr_cond[v]]
            lpm_means.append(s["lpm"].mean())
            lr_means.append(s["lr"].dropna().mean())
        ax_lpm.plot(vpr_order, lpm_means, marker="o", label=system)
        ax_lr.plot(vpr_order, lr_means, marker="o", label=system)
    ax_lpm.set_xlabel("VPR (dB)")
    ax_lpm.set_ylabel("Leaked notes / min")
    ax_lr.set_xlabel("VPR (dB)")
    ax_lr.set_ylabel("Voice-leak rate")
    ax_lpm.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_f3_piano_roll(ref_notes: list[dict], est_notes: list[dict], dur_s: float, out_path: Path) -> None:
    """F3: a piano-roll example. The clip is chosen in WP6 by a seeded rule; this is only the plotting function."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for n in ref_notes:
        ax.plot([n["onset_time"], n["offset_time"]], [n["midi_note"]] * 2, color="tab:blue", linewidth=3, alpha=0.6)
    for n in est_notes:
        ax.plot([n["onset_time"], n["offset_time"]], [n["midi_note"] + 0.3] * 2, color="tab:orange", linewidth=3, alpha=0.6)
    ax.set_xlim(0, dur_s)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI pitch")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
