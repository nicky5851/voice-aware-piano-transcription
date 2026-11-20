"""Loss terms (protocol §7.3, §7.4; spec WP3 §2.3).

total = L_kong + L_voice. lam never multiplies the loss -- it acts only
through grad_scale() on the voice head's input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

THIRD_PARTY_PYTORCH = Path(__file__).resolve().parents[3] / "third_party" / "piano_transcription" / "pytorch"


def _ensure_vendored_on_path() -> None:
    p = str(THIRD_PARTY_PYTORCH)
    if p not in sys.path:
        sys.path.insert(0, p)
    # losses.py has no further bare imports, so utils/ is not needed here.


def l_kong(output_dict: dict, target_dict: dict) -> dict:
    """Kong's vendored regress_onset_offset_frame_velocity_bce, unchanged, plus its 4 sub-terms."""
    _ensure_vendored_on_path()
    from losses import bce  # vendored, unchanged

    onset_loss = bce(output_dict["reg_onset_output"], target_dict["reg_onset_roll"], target_dict["mask_roll"])
    offset_loss = bce(output_dict["reg_offset_output"], target_dict["reg_offset_roll"], target_dict["mask_roll"])
    frame_loss = bce(output_dict["frame_output"], target_dict["frame_roll"], target_dict["mask_roll"])
    velocity_loss = bce(output_dict["velocity_output"], target_dict["velocity_roll"] / 128, target_dict["onset_roll"])
    total = onset_loss + offset_loss + frame_loss + velocity_loss
    return {
        "l_kong": total,
        "onset_loss": onset_loss,
        "offset_loss": offset_loss,
        "frame_loss": frame_loss,
        "velocity_loss": velocity_loss,
    }


def l_voice(voice_output: Tensor, voice_roll: Tensor, pos_weight: float = 1.0) -> Tensor:
    """Kong's clamped BCE form, averaged over B x T x 88 (dense target, no mask).

    pos_weight != 1 is only the protocol SS8.4 fallback (same value in every
    run once triggered); it is never used to rebalance lam.
    """
    eps = 1e-7
    output = torch.clamp(voice_output, eps, 1.0 - eps)
    matrix = -pos_weight * voice_roll * torch.log(output) - (1.0 - voice_roll) * torch.log(1.0 - output)
    return matrix.mean()


def total_loss(output_dict: dict, target_dict: dict, voice_roll: Tensor, pos_weight: float = 1.0) -> dict:
    kong = l_kong(output_dict, target_dict)
    voice = l_voice(output_dict["voice_output"], voice_roll, pos_weight=pos_weight)
    return {**kong, "l_voice": voice, "total": kong["l_kong"] + voice}
