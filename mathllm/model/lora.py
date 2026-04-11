"""Minimal LoRA (Low-Rank Adaptation) wrapper for linear layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LoRALinear(nn.Module):
    """Wraps a frozen linear layer with a low-rank trainable adapter.

    output = base_linear(x) + (alpha / rank) * x @ B.T @ A.T

    A is zero-initialized so LoRA starts as identity (no change to base output).
    B is initialized with small random values for symmetry breaking.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.scale = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_A = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: Tensor, gate: Tensor | None = None) -> Tensor:
        base_out = self.base_linear(x)
        # Compute LoRA path in float32 to avoid overflow on large vocab projections
        x_f = x.float()
        lora_out = (x_f @ self.lora_B.T) @ self.lora_A.T
        lora_contribution = self.scale * lora_out.to(dtype=base_out.dtype)
        if gate is not None:
            lora_contribution = lora_contribution * gate
        return base_out + lora_contribution
