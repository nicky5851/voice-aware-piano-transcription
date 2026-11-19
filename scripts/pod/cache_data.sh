#!/usr/bin/env bash
# Copy the HDF5/vocals/pitch files a job needs from the network volume onto
# local container disk, verified against a per-file sha256 manifest.
# Prints the time taken.
#
# ENGINEERING CHOICE: the manifest format is "<sha256>  <relative-path>"
# per line (sha256sum -c compatible), at
# manifests/checksums/<what>.sha256 relative to the shipped code directory.
# WP1 produces these manifests; this script only consumes them.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: cache_data.sh <what>" >&2
  exit 2
fi

what="$1"
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
manifest="$code_dir/manifests/checksums/${what}.sha256"
src_root="/workspace/vapt/work"
dst_root="${TMPDIR:-/root/cache}"

if [ ! -f "$manifest" ]; then
  echo "cache_data: no manifest at $manifest" >&2
  exit 1
fi

mkdir -p "$dst_root"
start_ts=$(date +%s)

while IFS= read -r line; do
  rel_path="${line#*  }"
  mkdir -p "$dst_root/$(dirname "$rel_path")"
  cp "$src_root/$rel_path" "$dst_root/$rel_path"
done < "$manifest"

cd "$dst_root"
if ! sha256sum -c "$manifest"; then
  echo "cache_data: checksum verification FAILED" >&2
  exit 1
fi

elapsed=$(( $(date +%s) - start_ts ))
echo "cache_data: cached '$what' to $dst_root in ${elapsed}s, verified against $manifest"
