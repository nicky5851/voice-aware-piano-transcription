#!/usr/bin/env bash
# Build (or verify) the locked virtual environment at /workspace/vapt/venv.
# Idempotent: a second run with an unchanged lock hash and image digest
# exits in under 10s without rebuilding.
set -euo pipefail

VENV_DIR="/workspace/vapt/venv"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK_FILE="$CODE_DIR/requirements.lock"

image_digest="${VAPT_IMAGE_DIGEST:-unknown}"
lock_hash="$(sha256sum "$LOCK_FILE" | cut -d' ' -f1)"
env_ok_contents="lock_hash=$lock_hash image_digest=$image_digest"

if [ -f "$VENV_DIR/ENV_OK" ] && [ "$(cat "$VENV_DIR/ENV_OK")" = "$env_ok_contents" ]; then
  echo "setup_env: venv already up to date ($VENV_DIR)"
  exit 0
fi

echo "setup_env: building venv at $VENV_DIR"
# Pinned to 3.11 (pyproject.toml's floor) regardless of the image's system
# Python, since RunPod base images have shipped 3.12+ as /usr/bin/python3.
uv venv --python 3.11 "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
uv pip sync "$LOCK_FILE"

echo "$env_ok_contents" > "$VENV_DIR/ENV_OK"
echo "setup_env: done"
