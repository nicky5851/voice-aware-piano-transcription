"""Gradient-scaling identity: the mechanism behind lambda (protocol §7.3)."""
from __future__ import annotations

import torch
from torch import Tensor


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h: Tensor, lam: float) -> Tensor:
        ctx.lam = lam
        return h

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output * ctx.lam, None


def grad_scale(h: Tensor, lam: float) -> Tensor:
    """Forward: identity. Backward: multiplies the incoming gradient by lam.

    lam == 0 uses h.detach() directly (an exact zero gradient with no autograd
    node at all, so there is nothing for a NaN upstream gradient to propagate
    through); lam == 1 returns h unchanged (ordinary gradient flow, no extra
    node); any other lam goes through the custom autograd function.
    """
    if lam == 0.0:
        return h.detach()
    if lam == 1.0:
        return h
    return _GradScale.apply(h, lam)
