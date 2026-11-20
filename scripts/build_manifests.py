#!/usr/bin/env python
"""CLI: build and checksum the validation and test manifests (spec WP2 §4).

No MIDI is read here: only HDF5 waveform audio (for non-silence) and the
voiced-window index (spec WP1). The test manifest's offsets are frozen now;
it is never rendered or scored until WP6.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vapt.mixing.manifests import build_manifest, write_manifest_with_checksum  # noqa: E402

WORKSPACE = Path("/workspace/vapt/work")


def main() -> None:
    hdf5s_dir = WORKSPACE / "hdf5s" / "maestro"
    repo_root = Path(__file__).resolve().parents[1]

    for which in ("val", "test"):
        windows_path = WORKSPACE / "pitch" / which / "voiced_windows_30s.json"
        voiced_windows_30s = json.loads(windows_path.read_text())
        entries = build_manifest(hdf5s_dir, voiced_windows_30s, which=which, seed=0)
        out_path = repo_root / "manifests" / f"{which}_manifest.json"
        write_manifest_with_checksum(entries, out_path)
        n_clips = len(entries) // len(set(e["condition"] for e in entries))
        n_fallback = sum(1 for e in entries if e["condition"] == "clean" and e["non_silent_fallback_flagged"])
        print(f"{which}_manifest.json: {n_clips} clips, {len(entries)} items, {n_fallback} non-silent fallbacks")


if __name__ == "__main__":
    main()
