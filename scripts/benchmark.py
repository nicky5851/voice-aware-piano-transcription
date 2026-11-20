"""WP4 §2.2 hardware benchmark: measured s/step, data-wait and peak memory at batch 12.

CLI: python -m scripts.benchmark --config configs/train/lambda1_seed0.yaml
     --precision fp32_strict --steps 250 --warmup 50 --out results/compute/benchmark_X.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from shutil import disk_usage

import torch

from vapt.constants import CLASSES_NUM, FRAMES_PER_SECOND
from vapt.infra.hardware import record_hardware
from vapt.models.voice_aware import VoiceAwareKong
from vapt.train import build_loader, load_config, seed_everything, set_tf32, training_step

LOCAL_DISK_MIN_GB = 50.0  # training cache headroom; engineering judgement, not protocol-fixed


def run_benchmark(
    config_path: Path,
    workspace: Path,
    precision: str,
    steps: int,
    warmup: int,
    device: torch.device,
    num_workers: int | None = None,
) -> dict:
    cfg = load_config(config_path)
    seed_everything(cfg.seed)
    set_tf32(precision)

    model = VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.optimizer["lr"], betas=tuple(cfg.optimizer["betas"]),
        eps=cfg.optimizer["eps"], weight_decay=cfg.optimizer["weight_decay"],
    )
    effective_workers = num_workers if num_workers is not None else cfg.hardware.get("num_workers", 0)
    _, loader = build_loader(cfg, workspace, start_step=0, num_workers=effective_workers)
    loader_iter = iter(loader)
    pos_weight = cfg.voice_loss["pos_weight"]

    def _step():
        data_t0 = time.time()
        batch = next(loader_iter)
        data_wait = time.time() - data_t0
        step_t0 = time.time()
        result = training_step(
            model, batch, cfg.lam, pos_weight, optimizer, device, autocast_bf16=precision == "bf16"
        )
        optimizer.step()
        return data_wait, time.time() - step_t0, result

    fits_batch_12 = True
    oom_reason = None
    try:
        for _ in range(warmup):
            _step()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        s_per_step, data_waits = [], []
        for _ in range(steps):
            dw, sps, _ = _step()
            data_waits.append(dw)
            s_per_step.append(sps)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            fits_batch_12 = False
            oom_reason = str(e)
            s_per_step, data_waits = [float("nan")], [float("nan")]
        else:
            raise

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" and fits_batch_12 else None
    total_mem_mb = torch.cuda.get_device_properties(device).total_memory / 1e6 if device.type == "cuda" else None
    vram_headroom_frac = (
        (total_mem_mb - peak_mem_mb) / total_mem_mb if (total_mem_mb and peak_mem_mb) else None
    )
    disk_free_gb = round(disk_usage(str(workspace)).free / 1e9, 1)

    mean_s_per_step = sum(s_per_step) / len(s_per_step)
    mean_data_wait = sum(data_waits) / len(data_waits)

    return {
        "config": str(config_path),
        "precision": precision,
        "steps": steps,
        "warmup": warmup,
        "fits_batch_12": fits_batch_12,
        "oom_reason": oom_reason,
        "s_per_step_mean": mean_s_per_step,
        "data_wait_s_mean": mean_data_wait,
        "data_wait_frac": (mean_data_wait / mean_s_per_step) if mean_s_per_step else None,
        "peak_memory_mb": peak_mem_mb,
        "total_memory_mb": total_mem_mb,
        "vram_headroom_frac": vram_headroom_frac,
        "local_disk_sufficient": disk_free_gb >= LOCAL_DISK_MIN_GB,
        "disk_free_gb": disk_free_gb,
        "hardware": record_hardware(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace/vapt/work"))
    parser.add_argument("--precision", choices=["fp32_strict", "tf32", "bf16"], default="fp32_strict")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = run_benchmark(
        args.config, args.workspace, args.precision, args.steps, args.warmup,
        torch.device(args.device), num_workers=args.num_workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "hardware"}, indent=2))


if __name__ == "__main__":
    main()
