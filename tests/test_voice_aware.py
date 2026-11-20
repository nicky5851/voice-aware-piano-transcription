import sys
from pathlib import Path

import pytest
import torch

from vapt.models.losses import l_voice
from vapt.models.voice_aware import VoiceAwareKong, load_pretrained

FRAMES_PER_SECOND = 100
CLASSES_NUM = 88
SEGMENT_S = 10.0
SAMPLE_RATE = 16000


def _model() -> VoiceAwareKong:
    torch.manual_seed(0)
    return VoiceAwareKong(FRAMES_PER_SECOND, CLASSES_NUM)


def _input(batch=1):
    torch.manual_seed(1)
    return torch.randn(batch, int(SEGMENT_S * SAMPLE_RATE))


def test_load_pretrained_only_voice_head_missing(tmp_path):
    third_party_pytorch = Path(__file__).resolve().parents[1] / "third_party" / "piano_transcription" / "pytorch"
    p = str(third_party_pytorch)
    if p not in sys.path:
        sys.path.insert(0, p)
    from models import Regress_onset_offset_frame_velocity_CRNN

    plain = Regress_onset_offset_frame_velocity_CRNN(FRAMES_PER_SECOND, CLASSES_NUM)
    checkpoint_path = tmp_path / "fake.pth"
    torch.save({"model": {f"note_model.{k}": v for k, v in plain.state_dict().items()}}, checkpoint_path)

    model = _model()
    result = load_pretrained(model, checkpoint_path)
    assert result["missing"] == sorted(["voice_head.1.weight", "voice_head.1.bias"])
    assert result["unexpected"] == []


def test_output_shapes_match_target_frame_count():
    model = _model()
    model.eval()
    x = _input(batch=1)
    with torch.no_grad():
        out = model(x, lam=1.0)
    expected_t = int(round(SEGMENT_S * FRAMES_PER_SECOND)) + 1  # Kong's TargetProcessor: +1
    for key in ("frame_output", "reg_onset_output", "reg_offset_output", "velocity_output", "voice_output"):
        assert out[key].shape == (1, expected_t, CLASSES_NUM), f"{key}: {out[key].shape}"


def _voice_loss_grad(model, x, lam, pos_weight=1.0):
    model.zero_grad(set_to_none=True)
    out = model(x, lam=lam)
    voice_roll = torch.zeros_like(out["voice_output"])
    loss = l_voice(out["voice_output"], voice_roll, pos_weight=pos_weight)
    loss.backward()
    grads = {name: (p.grad.clone() if p.grad is not None else None) for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return grads


@pytest.mark.parametrize("prefix", ["frame_model", "reg_onset_model"])
def test_lambda_zero_gives_exactly_zero_shared_gradient(prefix):
    model = _model()
    model.eval()
    x = _input()
    grads = _voice_loss_grad(model, x, lam=0.0)
    for name, g in grads.items():
        if name.startswith(prefix):
            assert g is None or torch.all(g == 0), f"{name} got nonzero gradient at lam=0: {g.abs().max()}"
    assert any(g is not None and torch.any(g != 0) for name, g in grads.items() if name.startswith("voice_head"))


def test_lambda_one_gives_nonzero_shared_gradient():
    model = _model()
    model.eval()
    x = _input()
    grads = _voice_loss_grad(model, x, lam=1.0)
    for prefix in ("frame_model", "reg_onset_model"):
        nonzero = any(g is not None and torch.any(g != 0) for name, g in grads.items() if name.startswith(prefix))
        assert nonzero, f"{prefix} got zero gradient from L_voice at lam=1"


def test_lambda_half_is_half_of_lambda_one_gradient():
    model = _model()
    model.eval()
    x = _input()
    grads_1 = _voice_loss_grad(model, x, lam=1.0)
    grads_half = _voice_loss_grad(model, x, lam=0.5)
    for prefix in ("frame_model", "reg_onset_model"):
        for name in grads_1:
            if not name.startswith(prefix):
                continue
            g1, gh = grads_1[name], grads_half[name]
            if g1 is None:
                continue
            expected = 0.5 * g1
            denom = expected.abs().max().clamp_min(1e-12)
            rel_err = (gh - expected).abs().max() / denom
            assert rel_err < 1e-6, f"{name}: rel_err={rel_err.item()}"


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_isolation_offset_velocity_and_refinement_grus_always_zero(lam):
    model = _model()
    model.eval()
    x = _input()
    grads = _voice_loss_grad(model, x, lam=lam)
    isolated_prefixes = ("reg_offset_model", "velocity_model", "reg_onset_gru", "reg_onset_fc", "frame_gru", "frame_fc")
    for name, g in grads.items():
        if any(name.startswith(p) for p in isolated_prefixes):
            assert g is None or torch.all(g == 0), f"{name} got nonzero gradient from L_voice at lam={lam}"
