#!/usr/bin/env bash
# Idle/completion watchdog, started on every pod. Stops the pod when either:
#   (a) every run under work/runs/ has DONE or FAILED, plus a 10-minute
#       grace period, or
#   (b) no heartbeat for 30 minutes and no interactive SSH session.
# Set WATCHDOG_DRY_RUN=1 to log the stop decision instead of acting on it.
set -euo pipefail

RUNS_DIR="/workspace/vapt/work/runs"
GRACE_SECONDS=600
IDLE_SECONDS=1800
POLL_SECONDS=60

log() { echo "watchdog: $(date -u +%FT%TZ) $*"; }

all_runs_finished() {
  local n_total=0 n_finished=0
  for run_dir in "$RUNS_DIR"/*/; do
    [ -d "$run_dir" ] || continue
    n_total=$((n_total + 1))
    if [ -f "${run_dir}DONE" ] || [ -f "${run_dir}FAILED" ]; then
      n_finished=$((n_finished + 1))
    fi
  done
  [ "$n_total" -gt 0 ] && [ "$n_total" -eq "$n_finished" ]
}

seconds_since_last_heartbeat() {
  local newest=0 mtime
  for hb in "$RUNS_DIR"/*/heartbeat.json; do
    [ -f "$hb" ] || continue
    mtime=$(stat -c %Y "$hb" 2>/dev/null || stat -f %m "$hb")
    [ "$mtime" -gt "$newest" ] && newest=$mtime
  done
  if [ "$newest" -eq 0 ]; then
    echo 999999
  else
    echo $(( $(date +%s) - newest ))
  fi
}

has_interactive_ssh() {
  who | grep -q pts
}

stop_pod() {
  local reason="$1"
  log "stop condition met: $reason"
  if [ "${WATCHDOG_DRY_RUN:-0}" = "1" ]; then
    log "dry run: would stop pod ${RUNPOD_POD_ID:-unknown} now"
    return 0
  fi
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    # rest.runpod.io does not resolve; the REST v2 host is api.runpod.io,
    # and the action endpoint is POST /v2/pods/{id}/action with a JSON
    # {"action": "stop"} body, not a /stop suffix (confirmed against the
    # live v2 OpenAPI doc and a real stop call, 2026-11-20).
    curl -s -X POST "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}/action" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -H "Content-Type: application/json" \
      -d '{"action":"stop"}' >/dev/null
    log "stop request sent for pod ${RUNPOD_POD_ID}"
  elif command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "${RUNPOD_POD_ID}"
    log "stop request sent via runpodctl for pod ${RUNPOD_POD_ID}"
  else
    log "ERROR: no credentials to self-stop (need RUNPOD_API_KEY+RUNPOD_POD_ID, or runpodctl+RUNPOD_POD_ID)"
    exit 1
  fi
  exit 0
}

grace_deadline=0
while true; do
  sleep "$POLL_SECONDS"

  if all_runs_finished; then
    if [ "$grace_deadline" -eq 0 ]; then
      grace_deadline=$(( $(date +%s) + GRACE_SECONDS ))
      log "all runs finished; starting ${GRACE_SECONDS}s grace period"
    elif [ "$(date +%s)" -ge "$grace_deadline" ]; then
      stop_pod "all runs DONE/FAILED, grace period elapsed"
    fi
  else
    grace_deadline=0
  fi

  idle_for=$(seconds_since_last_heartbeat)
  if [ "$idle_for" -ge "$IDLE_SECONDS" ] && ! has_interactive_ssh; then
    stop_pod "no heartbeat for ${idle_for}s and no interactive SSH session"
  fi
done
