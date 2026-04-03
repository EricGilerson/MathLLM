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
