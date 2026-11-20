"""Parallel wrapper around Kong's vendored MAESTRO packer (spec WP1 §4.1).

# ENGINEERING CHOICE: pack_maestro_dataset_to_hdf5 (vendored, unmodified)
# processes the whole CSV in one serial loop; the spec also asks to
# parallelise over files. Rather than edit the vendored function -- only
# allowed via a separate, recorded commit -- each worker gets its own
# chunk directory: a filtered copy of the CSV plus symlinks to just the
# audio/midi files it needs, and calls the unmodified vendored function on
# that chunk. Output HDF5 paths never collide across workers (one per
# source file), so no coordination beyond waiting for every worker.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

THIRD_PARTY_UTILS = Path(__file__).resolve().parents[3] / "third_party" / "piano_transcription" / "utils"


def _ensure_vendored_on_path() -> None:
    # The vendored features.py does `from utilities import ...` and
    # `import config` as bare imports (NOTICE documents this quirk).
    p = str(THIRD_PARTY_UTILS)
    if p not in sys.path:
        sys.path.insert(0, p)


def read_maestro_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _write_chunk(rows: list[dict[str, str]], fieldnames: list[str], chunk_dir: Path, dataset_dir: Path) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    with open(chunk_dir / "maestro-v2.0.0.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        for key in ("midi_filename", "audio_filename"):
            rel = row[key]
            dst = chunk_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(dataset_dir / rel)


def _pack_chunk(chunk_dir: str, workspace: str) -> None:
    _ensure_vendored_on_path()
    import features  # vendored, unmodified

    args = argparse.Namespace(dataset_dir=chunk_dir, workspace=workspace)
    features.pack_maestro_dataset_to_hdf5(args)


def pack_maestro_parallel(dataset_dir: Path, workspace: Path, num_workers: int = 4) -> dict[str, int]:
    """Packs every MAESTRO row to HDF5 via the vendored function, split across worker processes.

    Returns the per-split row counts, read from the CSV (never hard-coded).
    """
    csv_path = dataset_dir / "maestro-v2.0.0.csv"
    rows = read_maestro_rows(csv_path)
    fieldnames = list(rows[0].keys())

    chunks_root = workspace / "tmp_maestro_chunks"
    chunk_size = max(1, -(-len(rows) // num_workers))  # ceil division
    chunk_dirs = []
    for start in range(0, len(rows), chunk_size):
        chunk_rows = rows[start : start + chunk_size]
        chunk_dir = chunks_root / str(start // chunk_size)
        _write_chunk(chunk_rows, fieldnames, chunk_dir, dataset_dir)
        chunk_dirs.append(chunk_dir)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_pack_chunk, str(d), str(workspace)) for d in chunk_dirs]
        for fut in futures:
            fut.result()

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    return counts


class MaestroPackingTruncatedError(Exception):
    """Raised when fewer HDF5 files exist than the CSV's row count, e.g. a quota or worker crash."""


def verify_maestro_file_count(counts: dict[str, int], hdf5_dir: Path) -> int:
    """Hard assertion that packing produced one HDF5 file per CSV row (CLAUDE.md WP1 follow-up, 2026-11-20).

    pack_maestro_parallel's first run silently stopped partway through a quota limit (WP1 report:
    108 files missing, recovered only because counts were checked by hand); this makes that failure
    mode fail loudly instead, by comparing the CSV's row count against the files actually on disk.
    """
    expected = sum(counts.values())
    n_files = len(list(hdf5_dir.rglob("*.h5")))
    if n_files != expected:
        raise MaestroPackingTruncatedError(
            f"expected {expected} HDF5 files (from the CSV row count) but found {n_files} under {hdf5_dir}"
        )
    return n_files
