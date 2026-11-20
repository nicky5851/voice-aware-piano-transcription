"""sha256 checksums for data products (spec WP1 §3: one `checksums/<product>.sha256`)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(root: Path, files: list[Path], out_path: Path) -> None:
    """One `<sha256>  <path relative to root>` line per file, sorted for determinism."""
    lines = []
    for path in files:
        digest = sha256_file(path)
        rel = path.relative_to(root)
        lines.append(f"{digest}  {rel.as_posix()}")
    lines.sort(key=lambda line: line.split("  ", 1)[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
