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

import torch
import torch.nn as nn
from torch import Tensor

from mathllm.arb.stage1_extract import OperandExtractor
from mathllm.arb.stage2_encode import RNSCircleEncoder
from mathllm.arb.stage3_compute import ArithmeticCompute
from mathllm.arb.stage4_inject import ResultInjector

# Legacy default for GPT-2 ('=' is token 28)
_DEFAULT_EQ_TOKEN_ID = 28


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
        num_classes: int = 10,
        mlp_hidden: int = 128,
        use_attention: bool = False,
        attn_rank: int = 32,
        eq_token_id: int = _DEFAULT_EQ_TOKEN_ID,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_digits = num_digits
        self.num_results = num_results
        self.eq_token_id = eq_token_id

        # Injection receives decoded digit vectors, not raw circle encodings.
        # 5 operations: add (unsigned), sub (signed → digits + sign),
        # mul (unsigned), exp (unsigned), div (unsigned)
        # Sub has K+1 dims (K digits + sign bit), others have K dims each.
        # Total: 4*K + (K+1) = 5K + 1
        total_result_dim = num_results * num_digits + 1  # +1 for sub sign bit

        # Stage 1: Learned extraction
        self.extract = OperandExtractor(
            hidden_dim, num_digits,
            num_classes=num_classes,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            use_attention=use_attention,
            attn_rank=attn_rank,
        )

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

    def _decode_to_digits(
        self,
        a_circle: Tensor,
        b_circle: Tensor,
        b_exp_circle: Tensor,
    ) -> Tensor:
        """Compute all operations and decode results to digit vectors.

        Instead of passing raw circle encodings (cos/sin pairs) to the injector,
        we CRT-reconstruct each operation's result into an actual integer and
        decompose it into base-10 digits. This gives the injector clean, linearly
        separable features instead of opaque periodic encodings.

        Returns:
            [B, S, 5*K+1] — concatenated digit vectors for all 5 operations.
            Sub includes a sign bit. All values detached (no grad through CRT).
        """
        B, S = a_circle.shape[:2]
        device = a_circle.device

        # Compute all five operations in circle space
        add_c = self.compute.circle_add(a_circle, b_circle)
        sub_c = self.compute.circle_sub(a_circle, b_circle)
        mul_c = self.compute.circle_mul(a_circle, b_circle)
        exp_c = self.compute.circle_exp(a_circle, b_exp_circle)
        div_c = self.compute.circle_div(a_circle, b_circle)

        # CRT reconstruct to integers and decompose to digits.
        # This is frozen/detached — no gradients flow through CRT.
        with torch.no_grad():
            add_n = self.compute.crt_reconstruct(add_c)         # [B, S]
            sub_n = self.compute.crt_reconstruct_signed(sub_c)  # [B, S] signed
            mul_n = self.compute.crt_reconstruct(mul_c)
            exp_n = self.compute.crt_reconstruct(exp_c)
            div_n = self.compute.crt_reconstruct(div_c)

            add_digits = self.compute.integer_to_digits(add_n)                  # [B, S, K]
            sub_digits = self.compute.integer_to_digits_with_sign(sub_n)        # [B, S, K+1]
            mul_digits = self.compute.integer_to_digits(mul_n)
            exp_digits = self.compute.integer_to_digits(exp_n)
            div_digits = self.compute.integer_to_digits(div_n)

        # Move back to original device if CRT used CPU (MPS path)
        result = torch.cat(
            [add_digits, sub_digits, mul_digits, exp_digits, div_digits], dim=-1
        )
        return result.to(device=device, dtype=a_circle.dtype)

    def forward(
        self,
        h: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the full ARB on the hidden state.

        Args:
            h: [batch, seq_len, hidden_dim]
            input_ids: [batch, seq_len] token IDs for digit lookup
            attention_mask: [batch, seq_len] (1 = real, 0 = padding).

        Returns:
            h': [batch, seq_len, hidden_dim] — h + delta_h from arithmetic
            d_a: [batch, seq_len, num_digits] — extracted digit vector for operand A
            d_b: [batch, seq_len, num_digits] — extracted digit vector for operand B
        """
        # Stage 1: Deterministic extraction via token lookup
        d_a, d_b, _, _ = self.extract(h, input_ids, attention_mask)

        # Stage 2: Encode to RNS circles
        a_circle = self.encode(d_a)          # [B, S, m, 2]
        b_circle = self.encode(d_b)          # [B, S, m, 2]
        b_exp_circle = self.encode.encode_exponent(d_b)  # [B, S, m, 2]

        # Stage 3: Compute + CRT decode to digit vectors
        # [B, S, 5*K+1] — clean decimal digits, not circle encodings
        results = self._decode_to_digits(a_circle, b_circle, b_exp_circle)

        # Build injection mask: only inject at '=' position and after.
        B, S = input_ids.shape
        eq_present = (input_ids == self.eq_token_id)  # [B, S]
        has_eq = eq_present.any(dim=1)  # [B]
        eq_pos = torch.where(
            has_eq,
            eq_present.long().argmax(dim=1),
            torch.full((B,), S, device=input_ids.device),
        )  # [B]
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)  # [1, S]
        inject_mask = (positions >= eq_pos.unsqueeze(1)).float()  # [B, S]
        inject_mask = inject_mask.unsqueeze(2)  # [B, S, 1] for broadcasting

        # Stage 4: Inject decoded digits into hidden state
        h_prime = self.inject(results * inject_mask, h)
        return h_prime, d_a, d_b

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
