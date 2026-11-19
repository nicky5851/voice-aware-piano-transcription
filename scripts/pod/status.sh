#!/usr/bin/env bash
# One-line status per run: step/40000, s/step, ETA, last validation F1, GPU util.
# Reads work/runs/<name>/heartbeat.json, written by train.py (WP3).
set -euo pipefail

RUNS_DIR="/workspace/vapt/work/runs"
TOTAL_STEPS=40000

for run_dir in "$RUNS_DIR"/*/; do
  [ -d "$run_dir" ] || continue
  name="$(basename "$run_dir")"
  hb="${run_dir}heartbeat.json"
  if [ ! -f "$hb" ]; then
    echo "$name: no heartbeat yet"
    continue
  fi
  python3 - "$hb" "$name" "$TOTAL_STEPS" <<'PYEOF'
import json
import sys

hb_path, name, total_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(hb_path) as f:
    hb = json.load(f)

step = hb.get("step", 0)
s_per_step = hb.get("s_per_step")
val_f1 = hb.get("last_val_onset_f1")
gpu_util = hb.get("gpu_util_pct")

eta = "n/a"
if s_per_step:
    remaining_s = (total_steps - step) * s_per_step
    eta = f"{remaining_s / 3600:.1f}h"

print(
    f"{name}: step {step}/{total_steps} | "
    f"{s_per_step if s_per_step is not None else 'n/a'} s/step | "
    f"ETA {eta} | last val F1 {val_f1 if val_f1 is not None else 'n/a'} | "
    f"GPU {gpu_util if gpu_util is not None else 'n/a'}%"
)
PYEOF
done
