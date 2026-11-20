import torch

from vapt.models.losses import l_kong, l_voice, total_loss


def _targets(b=2, t=10, c=88):
    return {
        "reg_onset_roll": torch.rand(b, t, c),
        "reg_offset_roll": torch.rand(b, t, c),
        "frame_roll": torch.rand(b, t, c),
        "velocity_roll": torch.rand(b, t, c) * 128,
        "onset_roll": torch.randint(0, 2, (b, t, c)).float(),
        "mask_roll": torch.ones(b, t, c),
    }


def _outputs(b=2, t=10, c=88):
    return {
        "reg_onset_output": torch.rand(b, t, c),
        "reg_offset_output": torch.rand(b, t, c),
        "frame_output": torch.rand(b, t, c),
        "velocity_output": torch.rand(b, t, c),
        "voice_output": torch.rand(b, t, c),
    }


def test_l_kong_sums_four_subterms():
    outputs, targets = _outputs(), _targets()
    terms = l_kong(outputs, targets)
    total = terms["onset_loss"] + terms["offset_loss"] + terms["frame_loss"] + terms["velocity_loss"]
    assert torch.allclose(terms["l_kong"], total)


def test_l_voice_matches_clean_bce_when_pos_weight_one():
    voice_output = torch.rand(2, 10, 88)
    voice_roll = (torch.rand(2, 10, 88) > 0.9).float()
    got = l_voice(voice_output, voice_roll, pos_weight=1.0)
    expected = torch.nn.functional.binary_cross_entropy(
        torch.clamp(voice_output, 1e-7, 1 - 1e-7), voice_roll
    )
    assert torch.allclose(got, expected, atol=1e-6)


def test_l_voice_pos_weight_upweights_positive_term():
    voice_output = torch.full((1, 1, 2), 0.5)
    voice_roll = torch.tensor([[[1.0, 0.0]]])
    unweighted = l_voice(voice_output, voice_roll, pos_weight=1.0)
    weighted = l_voice(voice_output, voice_roll, pos_weight=20.0)
    assert weighted > unweighted


def test_total_loss_is_l_kong_plus_l_voice_and_lam_never_appears():
    import inspect

    outputs, targets = _outputs(), _targets()
    voice_roll = torch.rand(2, 10, 88)
    out = total_loss(outputs, targets, voice_roll)
    assert torch.allclose(out["total"], out["l_kong"] + out["l_voice"])
    for fn in (l_kong, l_voice, total_loss):
        assert "lam" not in inspect.signature(fn).parameters
