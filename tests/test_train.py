from pathlib import Path

import pytest
import torch
import yaml

from vapt.constants import CLASSES_NUM, FRAMES_PER_SECOND, SAMPLE_RATE
from vapt.models.voice_aware import VoiceAwareKong
from vapt.train import (
    TARGET_KEYS,
    assert_protocol_invariants,
    build_loader,
    load_checkpoint,
    load_config,
    run_diagnostics,
    save_checkpoint,
    training_step,
)

WORKSPACE = Path("/workspace/vapt/work")
_HAS_REAL_DATA = (
    (WORKSPACE / "hdf5s" / "maestro").exists()
    and (WORKSPACE / "pitch" / "train" / "voiced_windows_10s.json").exists()
)
needs_real_data = pytest.mark.skipif(not _HAS_REAL_DATA, reason="requires WP1 data on the pod's network volume")

DEVICE = torch.device("cpu")


def _base_raw_config(tmp_path: Path, **overrides) -> dict:
    raw = {
        "run_name": "test_run",
        "seed": 0,
        "lam": 1.0,
        "steps": 40000,
        "batch_size": 12,
        "optimizer": {"name": "adam", "lr": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0},
        "lr_decay": {"every": 5000, "factor": 0.9},
        "voice_loss": {"pos_weight": 1.0},
        "data": {"segment_s": 10.0, "vpr_range_db": [-9, 9], "p_clean": 0.2},
        "validation": {"every": 2000, "manifest": "manifests/val_manifest.json"},
        "checkpoint": {"every": 2000, "keep_last": 2},
        "diagnostics": {"every": 1000, "fixed_batch_seed": 9999},
        "hardware": {"precision": "fp32_strict", "num_workers": 0, "prefetch_factor": 2, "pin_memory": False, "cache_dir": str(tmp_path)},
    }
    raw.update(overrides)
    return raw


def _write_config(tmp_path: Path, name: str, **overrides) -> Path:
    raw = _base_raw_config(tmp_path, **overrides)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return path


def _tiny_model() -> VoiceAwareKong:
    torch.manual_seed(0)
    return VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM)


def _synthetic_batches(n: int, batch_size: int = 2, audio_seconds: float = 1.0) -> list[dict]:
    """Tiny (1s audio, small batch) synthetic batches, for CPU-light resume/diagnostics tests (spec WP3 SS6:
    'CPU deterministic mode, tiny model input'). These don't need real WP1 data or a protocol-sized batch."""
    torch.manual_seed(123)
    with torch.no_grad():
        probe = VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM)
        t_frames = probe(torch.zeros(1, int(audio_seconds * SAMPLE_RATE)), lam=1.0)["frame_output"].shape[1]
    batches = []
    for i in range(n):
        g = torch.Generator().manual_seed(1000 + i)
        batch = {"waveform": torch.randn(batch_size, int(audio_seconds * SAMPLE_RATE), generator=g)}
        for key in TARGET_KEYS:
            batch[key] = torch.rand(batch_size, t_frames, CLASSES_NUM, generator=g)
        batch["voice_roll"] = torch.rand(batch_size, t_frames, CLASSES_NUM, generator=g)
        batches.append(batch)
    return batches


# ------------------------------------------------------------------ config assertions (not real-data gated)


def test_assertions_refuse_non_protocol_batch_size():
    raw = _base_raw_config(Path("."), batch_size=6)
    with pytest.raises(AssertionError):
        assert_protocol_invariants(raw, allow_non_primary_steps=False)


def test_assertions_refuse_non_protocol_lambda():
    raw = _base_raw_config(Path("."), lam=0.3)
    with pytest.raises(AssertionError):
        assert_protocol_invariants(raw, allow_non_primary_steps=False)


def test_assertions_refuse_non_40000_steps_without_max_steps_override():
    raw = _base_raw_config(Path("."), steps=100)
    with pytest.raises(AssertionError):
        assert_protocol_invariants(raw, allow_non_primary_steps=False)
    assert_protocol_invariants(raw, allow_non_primary_steps=True)  # same config, allowed for a non-primary run


def test_load_config_marks_non_primary_when_max_steps_given(tmp_path):
    path = _write_config(tmp_path, "cfg.yaml", steps=40000)
    cfg = load_config(path, max_steps=10)
    assert cfg.non_primary is True
    assert cfg.steps == 10


# ------------------------------------------------------------------ same-data (requires real data; uses the actual stream)


@needs_real_data
def test_same_seed_lambda0_and_lambda1_see_identical_batches(tmp_path):
    cfg0 = load_config(_write_config(tmp_path, "l0.yaml", lam=0.0), max_steps=20)
    cfg1 = load_config(_write_config(tmp_path, "l1.yaml", lam=1.0), max_steps=20)

    _, loader0 = build_loader(cfg0, WORKSPACE, start_step=0)
    _, loader1 = build_loader(cfg1, WORKSPACE, start_step=0)

    from vapt.train import batch_waveform_hash

    hashes0 = [batch_waveform_hash(b["waveform"]) for _, b in zip(range(20), loader0)]
    hashes1 = [batch_waveform_hash(b["waveform"]) for _, b in zip(range(20), loader1)]
    assert hashes0 == hashes1
    assert len(set(hashes0)) > 1  # not trivially all-identical


# ------------------------------------------------------------------ resume (synthetic, tiny model input -- not real-data gated)


def test_resume_matches_uninterrupted_training(tmp_path):
    batches = _synthetic_batches(6)
    lam, pos_weight = 1.0, 1.0

    def _run(batches_subset, model, optimizer):
        for b in batches_subset:
            training_step(model, b, lam, pos_weight, optimizer, DEVICE)
            optimizer.step()

    # straight 6 steps
    torch.manual_seed(42)
    model_a = VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM).to(DEVICE)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1.0e-4)
    _run(batches, model_a, opt_a)

    # 3 steps + checkpoint + resume + 3 steps
    torch.manual_seed(42)
    model_b = VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM).to(DEVICE)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1.0e-4)
    sched_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=5000, gamma=0.9)
    _run(batches[:3], model_b, opt_b)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model_b, opt_b, sched_b, step=2, cfg_hash="h", git_sha=None)
    del model_b, opt_b, sched_b

    model_c = VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM).to(DEVICE)
    opt_c = torch.optim.Adam(model_c.parameters(), lr=1.0e-4)
    sched_c = torch.optim.lr_scheduler.StepLR(opt_c, step_size=5000, gamma=0.9)
    state = load_checkpoint(ckpt_path, model_c, opt_c, sched_c)
    assert state["step"] == 2
    _run(batches[3:], model_c, opt_c)

    for (n_a, p_a), (n_c, p_c) in zip(model_a.named_parameters(), model_c.named_parameters()):
        assert n_a == n_c
        assert torch.allclose(p_a, p_c, atol=1e-6), f"{n_a} diverged after resume"


# ------------------------------------------------------------------ diagnostics isolation (synthetic, tiny model input)


def test_diagnostics_do_not_change_training_trajectory():
    batches = _synthetic_batches(4)
    diag_batch = _synthetic_batches(1, batch_size=2)[0]
    lam, pos_weight = 0.5, 1.0

    def _run(with_diagnostics: bool, model, optimizer):
        for b in batches:
            training_step(model, b, lam, pos_weight, optimizer, DEVICE)
            if with_diagnostics:
                run_diagnostics(model, diag_batch, lam, pos_weight, DEVICE)
            optimizer.step()

    torch.manual_seed(7)
    model_no_diag = VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM).to(DEVICE)
    opt_no_diag = torch.optim.Adam(model_no_diag.parameters(), lr=1.0e-4)
    _run(False, model_no_diag, opt_no_diag)
    del opt_no_diag

    torch.manual_seed(7)
    model_diag = VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM).to(DEVICE)
    opt_diag = torch.optim.Adam(model_diag.parameters(), lr=1.0e-4)
    _run(True, model_diag, opt_diag)

    for (n1, p1), (n2, p2) in zip(model_no_diag.named_parameters(), model_diag.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=1e-6), f"{n1} diverged when diagnostics ran"


# ------------------------------------------------------------------ end-to-end smoke of run_training on a handful of steps


# run_training()'s full orchestration (checkpoint/heartbeat/validation/DONE-file wiring together,
# at the protocol's real batch_size=12 on real 10s audio) is exercised by the GPU smoke run (spec
# WP3 SS7), not here: at batch 12 Kong's own numbers put this around 29 GB, which is why the spec
# puts it on a GPU rather than in a CPU unit test. This CPU pod's 8 GB cgroup limit confirmed that
# directly (OOM-killed) when this was tried as a 2-step CPU test.
