import numpy as np
import pandas as pd

from vapt.eval.report import (
    figure_f1_onset_vs_vpr,
    figure_f2_leakage_vs_vpr,
    figure_f3_piano_roll,
    table_t1,
    table_t2,
    table_t3,
    write_table,
)
from vapt.eval.stats import aggregate, compute_primary_statistics, decide

CONDITIONS = ("clean", "-6", "0", "+6")


def _synthetic_df(n_clips=60, seeds=(0, 1, 2)) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    f1_means = {
        "P": {"clean": 90.0, "-6": 68.0, "0": 73.0, "+6": 63.0},
        "lam0": {"clean": 88.0, "-6": 82.0, "0": 83.0, "+6": 81.0},
        "lam1": {"clean": 88.3, "-6": 82.5, "0": 83.5, "+6": 81.5},
    }
    for ci in range(n_clips):
        clip_id = f"clip{ci:03d}"
        song = f"song{ci % 10}"
        for system in ("P", "lam0", "lam1"):
            system_seeds = (0,) if system == "P" else seeds
            for sd in system_seeds:
                for cond in CONDITIONS:
                    f1 = float(np.clip(f1_means[system][cond] + rng.normal(0, 1.5), 0, 100))
                    n_ref, n_est = 20, 19
                    n_match = min(n_est, int(round(n_est * (f1 / 100.0))))
                    n_fp = n_est - n_match
                    n_leak = int(rng.binomial(n_fp, 0.3)) if (cond != "clean" and n_fp > 0) else 0
                    rows.append(
                        {
                            "system": system, "seed": sd, "clip_id": clip_id, "musdb_song": song, "condition": cond,
                            "dur_s": 30.0, "n_ref": n_ref, "n_est": n_est, "n_match": n_match, "n_fp": n_fp,
                            "n_leak": n_leak, "n_leak_oct": n_leak,
                            "onset_p": f1, "onset_r": f1, "onset_f1": f1, "onoff_f1": f1,
                        }
                    )
    return pd.DataFrame(rows)


def test_tables_and_figures_run_end_to_end(tmp_path):
    df = _synthetic_df()
    agg = aggregate(df)

    t1 = table_t1(agg, systems=["P", "lam0", "lam1"])
    assert len(t1) == 3
    write_table(t1, tmp_path / "t1.csv")
    assert (tmp_path / "t1.csv").exists()
    assert (tmp_path / "t1.tex").exists()

    results = compute_primary_statistics(agg, B=500, seed=0)
    components, decisions = decide(results)
    t2 = table_t2(components, decisions)
    assert "hypothesis_pass" in t2.columns
    write_table(t2, tmp_path / "t2.csv")

    t3 = table_t3(agg, systems=["lam0", "lam1"], seeds=[0, 1, 2], df_with_seed=df)
    assert len(t3) == 6  # 2 systems x 3 seeds
    write_table(t3, tmp_path / "t3.csv")

    figure_f1_onset_vs_vpr(agg, ["P", "lam0", "lam1"], tmp_path / "f1.png")
    assert (tmp_path / "f1.png").exists()

    figure_f2_leakage_vs_vpr(agg, ["lam0", "lam1"], tmp_path / "f2.png")
    assert (tmp_path / "f2.png").exists()

    ref_notes = [{"midi_note": 60, "onset_time": 1.0, "offset_time": 1.5}]
    est_notes = [{"midi_note": 60, "onset_time": 1.0, "offset_time": 1.4}]
    figure_f3_piano_roll(ref_notes, est_notes, dur_s=10.0, out_path=tmp_path / "f3.png")
    assert (tmp_path / "f3.png").exists()
