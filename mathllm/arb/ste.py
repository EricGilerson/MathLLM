"""Straight-Through Estimator (STE) for non-differentiable rounding operations.

The ARB's Stage 1 rounds continuous projections to integer digit values.
round() has zero gradient almost everywhere. The STE passes gradients through
unchanged, treating round() as the identity in the backward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor


class _STERound(torch.autograd.Function):
    """Straight-through estimator for torch.round()."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> Tensor:
        return grad_output


class _STEClamp(torch.autograd.Function):
    """Straight-through estimator for torch.clamp()."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        low: float,
        high: float,
    ) -> Tensor:
        return torch.clamp(x, low, high)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> Tensor:
        return grad_output, None, None


def ste_round(x: Tensor) -> Tensor:
    """Round to nearest integer with straight-through gradient."""
    return _STERound.apply(x)


def ste_clamp(x: Tensor, low: float, high: float) -> Tensor:
    """Clamp with straight-through gradient."""
    return _STEClamp.apply(x, low, high)


def ste_round_clamp(x: Tensor, low: int = 0, high: int = 9) -> Tensor:
    """Round to nearest integer and clamp to [low, high] with straight-through gradient.

    Forward: round(clamp(x, low, high))
    Backward: identity (gradients pass through unchanged)
    """
    return ste_round(ste_clamp(x, float(low), float(high)))


def ste_argmax(logits: Tensor) -> Tensor:
    """Argmax on the last dimension with straight-through gradient.

    Forward: argmax(logits, dim=-1) — hard integer indices.
    Backward: gradients flow through a soft weighted-index approximation,
    i.e. (softmax(logits) * arange(C)).sum(-1).

    This enables gradient flow from downstream losses (e.g. LM loss in Phase 2+)
    back through the discrete digit selection to the classification logits.

    Args:
        logits: [..., C] classification logits

    Returns:
        [...] float tensor of integer indices in [0, C-1]
    """
    hard = logits.argmax(dim=-1).float()
    soft_weights = torch.softmax(logits, dim=-1)
    indices = torch.arange(logits.size(-1), device=logits.device, dtype=logits.dtype)
    soft = (soft_weights * indices).sum(dim=-1)
    # STE trick: forward uses hard, backward uses soft
    return hard - soft.detach() + soft
