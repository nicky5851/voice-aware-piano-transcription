"""Resumable training loop for the voice-aware model (protocol SS7, SS8.2; spec WP3 SS3).

CLI: python -m vapt.train --config configs/train/lambda1_seed0.yaml [--resume auto]
     [--max-steps N] [--pretrained PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm
import torch
import yaml
from torch.utils.data import DataLoader

from vapt.constants import CLASSES_NUM, FRAMES_PER_SECOND, SAMPLE_RATE
from vapt.eval.validate import score_manifest
from vapt.infer import transcribe
from vapt.infra.hardware import record_hardware
from vapt.mixing.stream import BatchSampler, ExampleStream, MixtureDataset, build_piano_segment_list
from vapt.models.losses import total_loss
from vapt.models.voice_aware import VoiceAwareKong, load_pretrained

# Protocol-fixed values (SS7.5); a run refuses to start if its config disagrees.
PROTOCOL_BATCH_SIZE = 12
PROTOCOL_STEPS = 40000
PROTOCOL_LR = 1.0e-4
PROTOCOL_BETAS = (0.9, 0.999)
PROTOCOL_EPS = 1.0e-8
PROTOCOL_WEIGHT_DECAY = 0.0
PROTOCOL_LR_DECAY_EVERY = 5000
PROTOCOL_LR_DECAY_FACTOR = 0.9
PROTOCOL_P_CLEAN = 0.2
PROTOCOL_VPR_RANGE = (-9.0, 9.0)
PROTOCOL_LAMBDAS = (0.0, 0.5, 1.0)

TARGET_KEYS = ("reg_onset_roll", "reg_offset_roll", "frame_roll", "velocity_roll", "onset_roll", "mask_roll")
SHARED_PREFIXES = ("frame_model", "reg_onset_model")
ISOLATED_PREFIXES = ("reg_offset_model", "velocity_model", "reg_onset_gru", "reg_onset_fc", "frame_gru", "frame_fc")
HEAD_PREFIX = "voice_head"
BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
VAL_SELECTION_CONDITIONS = ("-6", "0", "+6")


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    seed: int
    lam: float
    steps: int
    batch_size: int
    optimizer: dict
    lr_decay: dict
    voice_loss: dict
    data: dict
    validation: dict
    checkpoint: dict
    diagnostics: dict
    hardware: dict
    raw: dict
    non_primary: bool


def config_hash(raw: dict) -> str:
    return hashlib.sha1(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def assert_protocol_invariants(raw: dict, allow_non_primary_steps: bool) -> None:
    """Refuses to start if the config disagrees with the frozen protocol values (SS7.5)."""
    assert raw["batch_size"] == PROTOCOL_BATCH_SIZE, f"batch_size must be {PROTOCOL_BATCH_SIZE}, got {raw['batch_size']}"
    if not allow_non_primary_steps:
        assert raw["steps"] == PROTOCOL_STEPS, f"steps must be {PROTOCOL_STEPS}, got {raw['steps']} (pass --max-steps for a non-primary run)"

    opt = raw["optimizer"]
    assert opt["name"] == "adam", f"optimizer must be adam, got {opt['name']}"
    assert opt["lr"] == PROTOCOL_LR, f"lr must be {PROTOCOL_LR}, got {opt['lr']}"
    assert tuple(opt["betas"]) == PROTOCOL_BETAS, f"betas must be {PROTOCOL_BETAS}, got {opt['betas']}"
    assert opt["eps"] == PROTOCOL_EPS, f"eps must be {PROTOCOL_EPS}, got {opt['eps']}"
    assert opt["weight_decay"] == PROTOCOL_WEIGHT_DECAY, f"weight_decay must be {PROTOCOL_WEIGHT_DECAY}, got {opt['weight_decay']}"

    lr_decay = raw["lr_decay"]
    assert lr_decay["every"] == PROTOCOL_LR_DECAY_EVERY, f"lr_decay.every must be {PROTOCOL_LR_DECAY_EVERY}, got {lr_decay['every']}"
    assert lr_decay["factor"] == PROTOCOL_LR_DECAY_FACTOR, f"lr_decay.factor must be {PROTOCOL_LR_DECAY_FACTOR}, got {lr_decay['factor']}"

    data = raw["data"]
    assert data["p_clean"] == PROTOCOL_P_CLEAN, f"data.p_clean must be {PROTOCOL_P_CLEAN}, got {data['p_clean']}"
    assert tuple(data["vpr_range_db"]) == PROTOCOL_VPR_RANGE, f"data.vpr_range_db must be {PROTOCOL_VPR_RANGE}, got {data['vpr_range_db']}"

    assert raw["lam"] in PROTOCOL_LAMBDAS, f"lam must be one of {PROTOCOL_LAMBDAS}, got {raw['lam']}"


def load_config(path: Path, max_steps: int | None = None, run_name: str | None = None) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text())
    assert_protocol_invariants(raw, allow_non_primary_steps=max_steps is not None)
    return TrainConfig(
        run_name=run_name or raw["run_name"],
        seed=raw["seed"],
        lam=float(raw["lam"]),
        steps=max_steps if max_steps is not None else raw["steps"],
        batch_size=raw["batch_size"],
        optimizer=raw["optimizer"],
        lr_decay=raw["lr_decay"],
        voice_loss=raw["voice_loss"],
        data=raw["data"],
        validation=raw["validation"],
        checkpoint=raw["checkpoint"],
        diagnostics=raw["diagnostics"],
        hardware=raw["hardware"],
        raw=raw,
        non_primary=max_steps is not None,
    )


# --------------------------------------------------------------------------- setup


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def set_tf32(precision: str) -> None:
    allow = precision in ("tf32", "bf16")
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow


def _collate(batch: list[dict]) -> dict:
    out: dict[str, torch.Tensor] = {}
    for key in ("waveform", "voice_roll", *TARGET_KEYS):
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch])).float()
    return out


def build_loader(cfg: TrainConfig, workspace: Path, start_step: int, num_workers: int = 0) -> tuple[ExampleStream, DataLoader]:
    hdf5s_dir = workspace / "hdf5s" / "maestro"
    vocals16k_dir = workspace / "vocals16k"
    pitch_dir = workspace / "pitch"

    piano_segments = build_piano_segment_list(hdf5s_dir, split="train")
    voice_windows = json.loads((pitch_dir / "train" / "voiced_windows_10s.json").read_text())
    stream = ExampleStream(seed=cfg.seed, piano_segments=piano_segments, voice_windows=voice_windows, batch_size=cfg.batch_size)
    dataset = MixtureDataset(hdf5s_dir, vocals16k_dir, pitch_dir, stream, meter=pyloudnorm.Meter(SAMPLE_RATE))
    sampler = BatchSampler(stream, start_step=start_step)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_collate, num_workers=num_workers)
    return stream, loader


def batch_waveform_hash(waveform: torch.Tensor) -> str:
    return hashlib.sha1(waveform.cpu().numpy().tobytes()).hexdigest()


# --------------------------------------------------------------------------- step


def grad_norms_by_submodule(model: torch.nn.Module) -> dict[str, float]:
    sq_by_top: dict[str, float] = {}
    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        sq = float((p.grad.detach() ** 2).sum())
        total_sq += sq
        top = name.split(".")[0]
        sq_by_top[top] = sq_by_top.get(top, 0.0) + sq
    out = {k: v**0.5 for k, v in sq_by_top.items()}
    out["total"] = total_sq**0.5
    return out


def training_step(
    model: VoiceAwareKong, batch: dict, lam: float, pos_weight: float, optimizer: torch.optim.Optimizer, device: torch.device
) -> dict:
    waveform = batch["waveform"].to(device)
    voice_roll = batch["voice_roll"].to(device)
    target = {k: batch[k].to(device) for k in TARGET_KEYS}

    optimizer.zero_grad(set_to_none=True)
    out = model(waveform, lam=lam)
    losses = total_loss(out, target, voice_roll, pos_weight=pos_weight)
    losses["total"].backward()
    grad_norms = grad_norms_by_submodule(model)
    return {"losses": losses, "grad_norms": grad_norms}


# --------------------------------------------------------------------------- diagnostics (protocol SS8.2)


def _bn_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: buf.clone() for name, buf in model.state_dict().items() if name.endswith(BN_BUFFER_SUFFIXES)}


def _restore_bn_buffers(model: torch.nn.Module, saved: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    for name, buf in saved.items():
        state[name].copy_(buf)


def _grad_vector(model: torch.nn.Module, prefixes: tuple[str, ...]) -> torch.Tensor:
    parts = []
    for name, p in model.named_parameters():
        if any(name.startswith(pfx) for pfx in prefixes):
            parts.append(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel(), device=p.device))
    return torch.cat(parts) if parts else torch.zeros(0)


def run_diagnostics(model: VoiceAwareKong, batch: dict, lam: float, pos_weight: float, device: torch.device) -> dict:
    """Shared/isolated/head gradient norms for L_kong and L_voice, plus their cosine on the shared params.

    Isolated from the training trajectory (spec WP3 SS3 step 5): BatchNorm
    buffers and the caller's real gradients are saved and restored, and the
    whole pass runs inside torch.random.fork_rng so it consumes no RNG state
    that a real training step would otherwise have used.
    """
    device_ids = [device.index or 0] if device.type == "cuda" else []
    saved_grads = {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}
    bn_before = _bn_buffers(model)
    was_training = model.training

    with torch.random.fork_rng(devices=device_ids):
        model.train()
        waveform = batch["waveform"].to(device)
        voice_roll = batch["voice_roll"].to(device)
        target = {k: batch[k].to(device) for k in TARGET_KEYS}

        out = model(waveform, lam=lam)
        losses = total_loss(out, target, voice_roll, pos_weight=pos_weight)

        model.zero_grad(set_to_none=True)
        losses["l_kong"].backward(retain_graph=True)
        grad_kong_shared = _grad_vector(model, SHARED_PREFIXES)
        result = {
            "l_kong_shared_norm": float(grad_kong_shared.norm()),
            "l_kong_isolated_norm": float(_grad_vector(model, ISOLATED_PREFIXES).norm()),
            "l_kong_head_norm": float(_grad_vector(model, (HEAD_PREFIX,)).norm()),
        }

        model.zero_grad(set_to_none=True)
        losses["l_voice"].backward()
        grad_voice_shared = _grad_vector(model, SHARED_PREFIXES)
        result.update(
            {
                "l_voice_shared_norm": float(grad_voice_shared.norm()),
                "l_voice_isolated_norm": float(_grad_vector(model, ISOLATED_PREFIXES).norm()),
                "l_voice_head_norm": float(_grad_vector(model, (HEAD_PREFIX,)).norm()),
            }
        )

        denom = (grad_kong_shared.norm() * grad_voice_shared.norm()).clamp_min(1e-12)
        result["cosine_shared"] = float(torch.dot(grad_kong_shared, grad_voice_shared) / denom)
        result["loss_ratio_voice_over_kong"] = float(losses["l_voice"].detach() / losses["l_kong"].detach().clamp_min(1e-12))

    _restore_bn_buffers(model, bn_before)
    for name, p in model.named_parameters():
        p.grad = saved_grads[name].clone() if saved_grads[name] is not None else None
    model.train(was_training)
    return result


# --------------------------------------------------------------------------- checkpoint / heartbeat (WP0 SS7)


def _atomic_write(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def save_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, step: int, cfg_hash: str, git_sha: str | None
) -> None:
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config_hash": cfg_hash,
        "git_sha": git_sha,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    _atomic_write(path, lambda tmp: torch.save(state, tmp))


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler) -> dict:
    # weights_only=False: these are our own checkpoints, written by save_checkpoint in this same
    # process/run directory, never an untrusted download; they also carry python/numpy RNG state
    # objects that torch's default weights_only=True (since torch 2.6) refuses to unpickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.random.set_rng_state(state["torch_random_state"])
    if state["cuda_random_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_random_state"])
    return state


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return candidates[-1] if candidates else None


def prune_old_checkpoints(run_dir: Path, keep_last: int) -> None:
    candidates = sorted(run_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    for stale in candidates[:-keep_last] if keep_last > 0 else candidates:
        stale.unlink()


def write_heartbeat(run_dir: Path, step: int, losses: dict) -> None:
    payload = {"step": step, "time": time.time(), "losses": losses}
    _atomic_write(run_dir / "heartbeat.json", lambda tmp: tmp.write_text(json.dumps(payload)))


def write_status_file(run_dir: Path, name: str, content: str) -> None:
    (run_dir / name).write_text(content)


# --------------------------------------------------------------------------- validation (spec WP3 SS3 step 6)


def run_validation(
    model: VoiceAwareKong, entries: list[dict], hdf5s_dir: Path, vocals16k_dir: Path, pitch_dir: Path, device: torch.device
) -> list[dict]:
    was_training = model.training
    model.eval()

    def _transcribe(audio: np.ndarray) -> tuple[list[dict], np.ndarray]:
        return transcribe(model, audio, device=str(device))

    rows = score_manifest(_transcribe, entries, hdf5s_dir, vocals16k_dir, pitch_dir, musdb_split="val")
    model.train(was_training)
    return rows


def selection_metric(rows: list[dict]) -> float:
    """Protocol SS7's validation selection rule: mean onset F1 over {-6, 0, +6} dB."""
    vals = [r["onset_f1"] for r in rows if r["condition"] in VAL_SELECTION_CONDITIONS]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- orchestration


def run_training(
    cfg: TrainConfig,
    run_dir: Path,
    workspace: Path,
    device: torch.device,
    pretrained_path: Path | None = None,
    resume: bool = False,
    val_limit: int | None = None,
    num_workers: int = 0,
    git_sha: str | None = None,
    heartbeat_every_s: float = 60.0,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = config_hash(cfg.raw)

    seed_everything(cfg.seed)
    set_tf32(cfg.hardware.get("precision", "fp32_strict"))

    model = VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer["lr"],
        betas=tuple(cfg.optimizer["betas"]),
        eps=cfg.optimizer["eps"],
        weight_decay=cfg.optimizer["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay["every"], gamma=cfg.lr_decay["factor"])

    start_step = 0
    resumed_from = find_latest_checkpoint(run_dir) if resume else None
    if resumed_from is not None:
        state = load_checkpoint(resumed_from, model, optimizer, scheduler)
        assert state["config_hash"] == cfg_hash, "refusing to resume: config hash differs from this run's recorded one"
        assert state["git_sha"] == git_sha, "refusing to resume: git sha differs from this run's recorded one"
        start_step = state["step"] + 1
    elif pretrained_path is not None:
        load_pretrained(model, pretrained_path)

    hardware_record = record_hardware()
    (run_dir / "hardware.json").write_text(json.dumps(hardware_record))
    print(f"=== train.py run={cfg.run_name} seed={cfg.seed} lam={cfg.lam} start_step={start_step} config_hash={cfg_hash[:12]} ===")

    stream, loader = build_loader(cfg, workspace, start_step=start_step, num_workers=num_workers)
    diag_batch = None
    if cfg.diagnostics["every"]:
        diag_stream = ExampleStream(seed=cfg.diagnostics["fixed_batch_seed"], piano_segments=stream.piano_segments, voice_windows=stream.voice_windows, batch_size=cfg.batch_size)
        diag_dataset = MixtureDataset(workspace / "hdf5s" / "maestro", workspace / "vocals16k", workspace / "pitch", diag_stream, meter=pyloudnorm.Meter(SAMPLE_RATE))
        diag_batch = _collate([diag_dataset[spec] for spec in diag_stream.batch_specs(0)])

    batch_hashes_path = run_dir / "batch_hashes.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    pos_weight = cfg.voice_loss["pos_weight"]

    best_metric = -float("inf")
    last_heartbeat = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loader_iter = iter(loader)
    try:
        for step in range(start_step, cfg.steps):
            data_t0 = time.time()
            batch = next(loader_iter)
            data_wait_s = time.time() - data_t0
            step_t0 = time.time()

            if step < 200:
                with open(batch_hashes_path, "a") as f:
                    f.write(json.dumps({"step": step, "hash": batch_waveform_hash(batch["waveform"])}) + "\n")

            result = training_step(model, batch, cfg.lam, pos_weight, optimizer, device)
            losses = result["losses"]

            if not torch.isfinite(losses["total"]):
                save_checkpoint(run_dir / "debug_checkpoint.pt", model, optimizer, scheduler, step, cfg_hash, git_sha)
                write_status_file(run_dir, "FAILED", f"non-finite loss at step {step}: {losses['total'].detach().item()}")
                return

            optimizer.step()
            scheduler.step()

            if cfg.diagnostics["every"] and step % cfg.diagnostics["every"] == 0:
                run_diagnostics(model, diag_batch, cfg.lam, pos_weight, device)

            now = time.time()
            if now - last_heartbeat >= heartbeat_every_s:
                write_heartbeat(run_dir, step, {k: float(v.detach()) for k, v in losses.items()})
                last_heartbeat = now

            if step % 50 == 0:
                log = {
                    "step": step,
                    "lr": scheduler.get_last_lr()[0],
                    "losses": {k: float(v.detach()) for k, v in losses.items()},
                    "grad_norms": result["grad_norms"],
                    "s_per_step": now - step_t0,
                    "data_wait_s": data_wait_s,
                    "gpu_mem_mb": torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(log) + "\n")

            if cfg.checkpoint["every"] and step % cfg.checkpoint["every"] == 0:
                save_checkpoint(run_dir / f"checkpoint_step_{step}.pt", model, optimizer, scheduler, step, cfg_hash, git_sha)
                prune_old_checkpoints(run_dir, cfg.checkpoint["keep_last"])

            if cfg.validation["every"] and step % cfg.validation["every"] == 0:
                entries = json.loads(Path(cfg.validation["manifest"]).read_text())
                if val_limit is not None:
                    entries = entries[:val_limit]
                rows = run_validation(model, entries, workspace / "hdf5s" / "maestro", workspace / "vocals16k", workspace / "pitch", device)
                metric = selection_metric(rows)
                (run_dir / "val").mkdir(exist_ok=True)
                (run_dir / "val" / f"step_{step}.json").write_text(json.dumps({"step": step, "selection_metric": metric, "rows": rows}))
                if metric > best_metric:
                    best_metric = metric
                    save_checkpoint(run_dir / "best.pth", model, optimizer, scheduler, step, cfg_hash, git_sha)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            write_status_file(run_dir, "FAILED", f"OOM at batch {cfg.batch_size}")
            return
        raise

    write_status_file(run_dir, "DONE", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", choices=["auto"], default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace/vapt/work"))
    parser.add_argument("--runs-dir", type=Path, default=Path("/workspace/vapt/work/runs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, max_steps=args.max_steps, run_name=args.run_name)
    git_sha_path = Path("GIT_SHA")
    git_sha = git_sha_path.read_text().strip() if git_sha_path.exists() else None

    run_training(
        cfg=cfg,
        run_dir=args.runs_dir / cfg.run_name,
        workspace=args.workspace,
        device=torch.device(args.device),
        pretrained_path=args.pretrained,
        resume=args.resume == "auto",
        val_limit=args.val_limit,
        num_workers=args.num_workers if args.num_workers is not None else cfg.hardware.get("num_workers", 0),
        git_sha=git_sha,
    )


if __name__ == "__main__":
    main()
