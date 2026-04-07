"""Arithmetic Residual Block: the complete ARB module.

Combines all four stages:
  1. Extract (learned) — read digit vectors from hidden state
  2. Encode (frozen) — map to RNS circle representation
  3. Compute (frozen) — execute +, -, *, exp, / in parallel
  4. Inject (learned) — project results back into hidden state

The ARB runs unconditionally on every token, like LayerNorm. Non-math tokens
receive near-zero contribution because the injection projection learns to
suppress irrelevant outputs.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from mathllm.arb.stage1_extract import OperandExtractor
from mathllm.arb.stage2_encode import RNSCircleEncoder
from mathllm.arb.stage3_compute import ArithmeticCompute
from mathllm.arb.stage4_inject import ResultInjector


class ArithmeticResidualBlock(nn.Module):
    """Complete Arithmetic Residual Block.

    Learned parameters: ~47K for GPT-2 (d=768, K=10)
    Frozen parameters: ~25K (tables, coefficient matrices, CRT weights)
    Compute cost: negligible (~54 FLOPs for add/sub per token)
    """

    def __init__(
        self,
        hidden_dim: int,
        primes: tuple[int, ...] = (7, 11, 13, 17, 19, 23, 29, 31, 37),
        num_digits: int = 10,
        num_results: int = 5,
        softmax_temperature: float = 1000.0,
        dropout: float = 0.1,
        injector_init_std: float = 1e-3,
        gate_init_logit: float = -2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_digits = num_digits
        self.num_results = num_results

        # Each operation outputs m primes * 2 (cos, sin) = 2m dimensions
        num_primes = len(primes)
        total_result_dim = num_results * num_primes * 2

        # Stage 1: Learned extraction
        self.extract = OperandExtractor(hidden_dim, num_digits, dropout=dropout)

        # Stage 2: Frozen encoding
        self.encode = RNSCircleEncoder(primes, num_digits)

        # Stage 3: Frozen computation
        self.compute = ArithmeticCompute(
            primes,
            num_digits,
            softmax_temperature,
            repair_division_during_training=False,
        )

        # Stage 4: Learned injection
        self.inject = ResultInjector(
            hidden_dim,
            total_result_dim,
            dropout=dropout,
            init_std=injector_init_std,
            gate_init_logit=gate_init_logit,
        )

        # Freeze stages 2 and 3
        for param in self.encode.parameters():
            param.requires_grad = False
        for param in self.compute.parameters():
            param.requires_grad = False

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run the full ARB on the hidden state.

        Args:
            h: [batch, seq_len, hidden_dim]

        Returns:
            h': [batch, seq_len, hidden_dim] — h + delta_h from arithmetic
            d_a: [batch, seq_len, num_digits] — extracted digit vector for operand A
            d_b: [batch, seq_len, num_digits] — extracted digit vector for operand B
            d_a_cont: [batch, seq_len, num_digits] — continuous (pre-round) operand A
            d_b_cont: [batch, seq_len, num_digits] — continuous (pre-round) operand B
        """
        # Stage 1: Extract operand digit vectors
        d_a, d_b, d_a_cont, d_b_cont = self.extract(h)

        # Stage 2: Encode to RNS circles (uses rounded values)
        a_circle = self.encode(d_a)          # [B, S, m, 2]
        b_circle = self.encode(d_b)          # [B, S, m, 2]
        b_exp_circle = self.encode.encode_exponent(d_b)  # [B, S, m, 2]

        # Stage 3: Compute all operations in parallel
        # Returns flattened circle encodings: [B, S, 5 * m * 2]
        results = self.compute(a_circle, b_circle, b_exp_circle)

        # Stage 4: Inject into hidden state with residual connection
        h_prime = self.inject(results, h)
        return h_prime, d_a, d_b, d_a_cont, d_b_cont

    def count_parameters(self) -> dict[str, int]:
        """Count learned vs frozen parameters."""
        learned = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        frozen = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        # Also count buffers (frozen tensors registered via register_buffer)
        buffers = sum(b.numel() for b in self.buffers())
        return {
            "learned": learned,
            "frozen_params": frozen,
            "frozen_buffers": buffers,
            "total": learned + frozen + buffers,
        }
