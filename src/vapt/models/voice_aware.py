"""The voice-aware Kong model (protocol SS7.1-7.2; spec WP3 SS2.2).

Piano outputs are computed exactly as Kong's own forward pass; only the
voice head and its gradient-scaled input are new. Vendored reference:
third_party/piano_transcription/pytorch/models.py (pinned commit, see
third_party/piano_transcription/NOTICE), classes AcousticModelCRnn8Dropout
and Regress_onset_offset_frame_velocity_CRNN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vapt.models.grad_scale import grad_scale

THIRD_PARTY_PYTORCH = Path(__file__).resolve().parents[3] / "third_party" / "piano_transcription" / "pytorch"
THIRD_PARTY_UTILS = Path(__file__).resolve().parents[3] / "third_party" / "piano_transcription" / "utils"


def _ensure_vendored_on_path() -> None:
    for p in (str(THIRD_PARTY_PYTORCH), str(THIRD_PARTY_UTILS)):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_vendored_on_path()
from models import (  # noqa: E402  (vendored, unmodified)
    AcousticModelCRnn8Dropout,
    Regress_onset_offset_frame_velocity_CRNN,
    init_layer,
)


class AcousticModelWithFeatures(AcousticModelCRnn8Dropout):
    """Same forward pass as the parent; also returns the BiGRU output before
    the final dropout and fc, as [B, T, 512]. Parameter names are unchanged
    (no new modules), so a Kong checkpoint loads onto this class unmodified.
    Body copied verbatim from AcousticModelCRnn8Dropout.forward (see module
    docstring for the vendored source), not re-derived.
    """

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_block1(input, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)

        (x, _) = self.gru(x)
        features = x  # [B, T, 512], before the final dropout and fc
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))
        return output, features


class VoiceAwareKong(Regress_onset_offset_frame_velocity_CRNN):
    """Kong's transcriber plus a gradient-scaled singer-pitch auxiliary head.

    The voice head is always trained with the full voice loss; lam (passed
    at forward time, never stored) controls only how much of that loss's
    gradient reaches the shared frame/onset features, via grad_scale().
    """

    def __init__(self, frames_per_second: int, classes_num: int):
        super().__init__(frames_per_second, classes_num)
        midfeat = 1792
        momentum = 0.01
        # Same constructor arguments as the parent used for these two sub-networks.
        self.frame_model = AcousticModelWithFeatures(classes_num, midfeat, momentum)
        self.reg_onset_model = AcousticModelWithFeatures(classes_num, midfeat, momentum)

        self.voice_head = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, 88))
        init_layer(self.voice_head[1])

    def forward(self, input: Tensor, lam: float, return_features: bool = False) -> dict:
        """Body copied from Regress_onset_offset_frame_velocity_CRNN.forward (see module
        docstring), with frame_model/reg_onset_model now also yielding their features,
        and the voice head appended. Do not re-derive the piano path from scratch.
        """
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        frame_output, frame_feat = self.frame_model(x)
        reg_onset_output, onset_feat = self.reg_onset_model(x)
        reg_offset_output = self.reg_offset_model(x)
        velocity_output = self.velocity_model(x)

        y = torch.cat((reg_onset_output, (reg_onset_output**0.5) * velocity_output.detach()), dim=2)
        (y, _) = self.reg_onset_gru(y)
        y = F.dropout(y, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(y))

        y = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (y, _) = self.frame_gru(y)
        y = F.dropout(y, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(y))

        head_input = grad_scale(torch.cat([frame_feat, onset_feat], dim=-1), lam)
        voice_output = torch.sigmoid(self.voice_head(head_input))

        output_dict = {
            "reg_onset_output": reg_onset_output,
            "reg_offset_output": reg_offset_output,
            "frame_output": frame_output,
            "velocity_output": velocity_output,
            "voice_output": voice_output,
        }
        if return_features:
            output_dict["frame_feat"] = frame_feat
            output_dict["onset_feat"] = onset_feat
        return output_dict


def load_pretrained(model: nn.Module, path: str | Path) -> dict:
    """Loads Kong's released note_model weights; only the voice head is left randomly initialised."""
    checkpoint = torch.load(path, map_location="cpu")
    prefix = "note_model."
    state = {k[len(prefix) :]: v for k, v in checkpoint["model"].items() if k.startswith(prefix)}
    result = model.load_state_dict(state, strict=False)

    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    expected_missing = {"voice_head.1.weight", "voice_head.1.bias"}
    if missing != expected_missing:
        raise AssertionError(f"missing keys {missing} != expected {expected_missing}")
    if unexpected:
        raise AssertionError(f"unexpected keys in checkpoint: {unexpected}")
    return {"missing": sorted(missing), "unexpected": sorted(unexpected)}
