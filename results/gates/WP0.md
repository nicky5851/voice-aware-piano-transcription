# WP0 gate report: RunPod compute strategy, environment and repository bootstrap

**Status: G0 passed**, with one item deferred (not failed) to WP4: the
GPU-side `record_hardware()` capture. The spec's G0 criterion asks for a
capture from one CPU pod and one GPU pod; only the CPU pod was done here.
It will be captured on WP4's first GPU pod, and WP4's gate report will
confirm it.

## Acceptance criteria

| # | Criterion | Result |
|---|---|---|
| 1 | RunPod access routes documented; SSH to a test pod works | Pass. RunPod MCP server used for all control-plane operations (documented in `runpod/README.md`, workspace-internal). SSH confirmed: the proxy address (`ssh.runpod.io`) only opens an interactive shell and cannot run non-interactive commands; the pod's **direct** address (public IP:port) is a normal sshd and was confirmed working for exec, rsync-style transfer, and tmux-detached jobs. |
| 2 | Network volume created in the recorded data centre; survey saved | Pass. 300GB volume in `EU-RO-1`, chosen by GPU-stock survey (`docs/compute.md`; full survey data kept with internal records). |
| 3 | `setup_env.sh` builds the locked venv; `ENV_OK` written; `record_hardware()` saved from one CPU pod and one GPU pod | Partial. CPU-pod capture done (`results/compute/wp0_cpu_pod.json`); idempotency confirmed (second run: 0.016s). GPU-pod capture deferred to WP4 (see above). |
| 4 | `ship.sh` → `run.sh` → `logs.sh` round trip works; the job survives an SSH disconnect | Pass. Verified live: a tmux-detached job kept running and completed correctly after its controlling SSH/tail connection was killed mid-job; the finished log was readable over a fresh connection. |
| 5 | Watchdog self-stop is verified, or its absence is documented and polling is in place | Pass (verified, not absent). `RUNPOD_API_KEY`/`RUNPOD_POD_ID` are present on pods by default. Both the dry-run paths (idle timeout; all-runs-finished + grace period) and one real self-stop were exercised live; the pod transitioned to `EXITED` via its own credentials. |
| 6 | Repo created privately; hooks installed; `prereg-v3` tag pushed; vendored Kong code with LICENSE and NOTICE | Pass. Hooks installed and active (pre-commit/commit-msg/pre-push). `prereg-v3` tag pushed to `origin` (confirmed via `git ls-remote --tags`). Kong's code vendored at a pinned commit with LICENSE and NOTICE. |
| 7 | `pytest -q` passes on a pod | Pass. 7 passed, 1 skipped (a GPU-marked test, correctly inapplicable on a CPU pod). |
| 8 | `runpod/ledger.csv` shows every pod used, and all of them are stopped | Pass. All pods used this work package are stopped/terminated; none remain on the account. |
| 9 | `walkthroughs/WP0.md` answers §12 | Pass (workspace-internal, not in this repository). |

## Bugs found and fixed during verification

- **Hardware fingerprinting leaked the host's totals, not the container's allocation.** `record_hardware()`'s CPU-count and RAM readings used `os.cpu_count()` and `/proc/meminfo`, which report a RunPod container's *host* machine, not its cgroup-limited share. Fixed to use `os.sched_getaffinity(0)` and `/sys/fs/cgroup/memory.max`. This affects every run's reproducibility record, not just WP0's.
- **Watchdog self-stop used a non-resolving API host.** `scripts/pod/watchdog.sh` called `rest.runpod.io/v1/pods/{id}/stop`, which fails DNS resolution, and even the corrected host's path was wrong (`/stop` instead of `/action` with a `{"action":"stop"}` body). Fixed and verified with a real stop call from inside a pod.
- **A job-runner script could silently drop output.** The dev workspace's `run.sh` (code-shipping helper, not part of this repository) redirected a multi-statement job's output to its log file in a way that only captured the last `;`-separated statement; fixed to wrap the whole command in a `{ ; }` group before redirecting.

## Spend

Total WP0 spend: **$0.15** of the $25 cap approved for WP0 preparation/tests. All pods were CPU-only; no GPU pod has been started yet. `runpod/ledger.csv` (workspace-internal) has the full per-pod breakdown.
