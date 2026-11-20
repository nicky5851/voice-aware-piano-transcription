"""Hardware and environment fingerprinting for reproducibility (protocol SS10.4)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import disk_usage
from typing import Any


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _git_sha() -> str | None:
    code_dir = os.environ.get("VAPT_CODE_DIR")
    search_roots = [Path(code_dir)] if code_dir else []
    search_roots += [Path.cwd(), *Path.cwd().parents]
    for root in search_roots:
        candidate = root / "GIT_SHA"
        if candidate.exists():
            return candidate.read_text().strip()
    return None


def _disk_free_gb(path: str = "/workspace") -> float | None:
    try:
        return round(disk_usage(path).free / 1e9, 1)
    except OSError:
        return None


def _cpu_count() -> int | None:
    # os.cpu_count() reports the host's total CPUs, not a container's
    # cgroup-limited allocation (seen on RunPod pods); affinity does.
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"count": _cpu_count(), "model": None}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def _ram_gb() -> float | None:
    # /proc/meminfo reports the host's total RAM, not a container's
    # cgroup-limited allocation (seen on RunPod pods); cgroup v2's
    # memory.max does, when the container actually has a limit set.
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        if limit != "max":
            return round(int(limit) / 1e9, 1)
    except (OSError, ValueError):
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1e6, 1)
    except OSError:
        pass
    return None


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"name": None, "count": 0, "memory_gb": None, "driver": None}
    try:
        import torch
    except ImportError:
        return info
    info["count"] = torch.cuda.device_count()
    if info["count"] > 0:
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["memory_gb"] = round(props.total_memory / 1e9, 1)
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if driver:
        info["driver"] = driver.splitlines()[0]
    return info


def record_hardware() -> dict[str, Any]:
    """Collect a reproducibility fingerprint of the current machine.

    Covers GPU name/count/VRAM/driver, CUDA/cuDNN/torch versions and TF32
    flags, CPU model/count, RAM, disk free, the container image digest and
    RunPod pod/GPU-type ids (from env, if present), and the git sha of the
    shipped code (from `<code_dir>/GIT_SHA`). Runs on CPU-only machines too
    (GPU and torch fields stay None). Every script calls this at start and
    writes the result next to its outputs.
    """
    record: dict[str, Any] = {
        "gpu": _gpu_info(),
        "cpu": _cpu_info(),
        "ram_gb": _ram_gb(),
        "disk_free_gb": _disk_free_gb(),
        "image_digest": os.environ.get("VAPT_IMAGE_DIGEST"),
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "runpod_gpu_type_id": os.environ.get("RUNPOD_GPU_TYPE_ID"),
        "git_sha": _git_sha(),
        "cuda_version": None,
        "cudnn_version": None,
        "torch_version": None,
        "tf32_matmul": None,
        "tf32_cudnn": None,
    }
    try:
        import torch

        record["torch_version"] = torch.__version__
        record["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            record["cudnn_version"] = torch.backends.cudnn.version()
        record["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        record["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    except ImportError:
        pass
    return record
