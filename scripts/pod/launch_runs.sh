#!/usr/bin/env bash
# Start one training process per entry in a launch plan (YAML: a `runs`
# list of {name, gpu, cmd}), each in its own tmux session with
# CUDA_VISIBLE_DEVICES set, logging to work/logs/<name>.log.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: launch_runs.sh <plan.yaml>" >&2
  exit 2
fi

plan="$1"
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
log_dir="/workspace/vapt/work/logs"
mkdir -p "$log_dir"

while IFS=$'\t' read -r name gpu cmd; do
  tmux new -d -s "$name" \
    "cd '$code_dir' && CUDA_VISIBLE_DEVICES=$gpu $cmd >> '$log_dir/$name.log' 2>&1"
  echo "launch_runs: started '$name' on GPU $gpu (log: $log_dir/$name.log)"
done < <(python3 - "$plan" <<'PYEOF'
import sys
import yaml

with open(sys.argv[1]) as f:
    plan = yaml.safe_load(f)

for run in plan["runs"]:
    print(f"{run['name']}\t{run['gpu']}\t{run['cmd']}")
PYEOF
)
