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
        injection_pos_dim: int = 0,
        injection_mlp_hidden: int = 0,
    ):
        super().__init__()

        # Generation cache state (used to fix KV cache bug)
        self._generation_mode = False
        self._cached_results: Tensor | None = None
        self._cached_has_eq: Tensor | None = None
        self._generation_offset = 0
        self.hidden_dim = hidden_dim
        self.num_digits = num_digits
        self.num_results = num_results
        self.eq_token_id = eq_token_id
        self.injection_pos_dim = injection_pos_dim

        # Injection receives a single selected result: K digits + 1 sign bit.
        # Operation selection (in _select_operation_result) picks the result
        # matching the detected operator and places it at fixed positions,
        # so the injection MLP always reads the same structure.
        digit_result_dim = num_digits + 1  # K digits + sign bit

        # Position-aware injection: add learned answer-position embeddings
        # so each answer position gets a unique injection signal.
        _MAX_ANSWER_TOKENS = 16  # enough for any answer length
        if injection_pos_dim > 0:
            self.answer_pos_embed = nn.Embedding(_MAX_ANSWER_TOKENS, injection_pos_dim)
            total_result_dim = digit_result_dim + injection_pos_dim
        else:
            self.answer_pos_embed = None
            total_result_dim = digit_result_dim

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
            mlp_hidden=injection_mlp_hidden,
        )

        # Freeze stages 2 and 3
        for param in self.encode.parameters():
            param.requires_grad = False
        for param in self.compute.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Generation cache mode
    # ------------------------------------------------------------------

    def enter_generation_mode(self) -> None:
        """Enable generation cache mode.

        On the first forward pass, arithmetic results are cached.
        Subsequent passes reuse the cached results with updated
        position offsets, fixing the KV cache bug where step 2+
        receives only the last token and can't find the operator.
        """
        self._generation_mode = True
        self._cached_results = None
        self._cached_has_eq = None
        self._generation_offset = 0

    def exit_generation_mode(self) -> None:
        """Disable generation cache mode and clear cached state."""
        self._generation_mode = False
        self._cached_results = None
        self._cached_has_eq = None
        self._generation_offset = 0

    def _forward_generation_cached(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass during generation using cached arithmetic results.

        Called on step 2+ when only the last generated token is passed.
        Uses cached digit results from the prompt pass with a fresh
        position embedding for the current answer offset.
        """
        B = h.shape[0]
        K = self.num_digits
        device = h.device

        # Expand cached results to [B, 1, D]
        results = self._cached_results.unsqueeze(1)

        # Add position embedding for current answer offset
        if self.answer_pos_embed is not None:
            offset = min(self._generation_offset, 15)
            offset_tensor = torch.full((B, 1), offset, dtype=torch.long, device=device)
            pos_emb = self.answer_pos_embed(offset_tensor)  # [B, 1, pos_dim]
            results = torch.cat([results, pos_emb], dim=-1)

        self._generation_offset += 1

        # Inject into hidden state
        h_prime = self.inject(results, h)

        # Apply cached sequence-level gate (suppress for non-arithmetic prompts)
        if self._cached_has_eq is not None:
            seq_gate = self._cached_has_eq.float().view(B, 1, 1)
            h_prime = h + (h_prime - h) * seq_gate

        # Dummy extractions (not used during generation)
        d_a = torch.zeros(B, 1, K, device=device)
        d_b = torch.zeros(B, 1, K, device=device)
        return h_prime, d_a, d_b

    def prepare_for_device(self, device: torch.device | str) -> None:
        """Warm runtime-only frozen buffers after the module is moved."""
        self.compute.prepare_for_device(device)

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

    def _select_operation_result(self, results: Tensor, input_ids: Tensor) -> Tensor:
        """Extract the selected operation's digits into a fixed-position vector.

        The ARB computes all 5 operations in parallel, but only one is relevant.
        This extracts that result and places it at fixed positions [0:K+1]
        (K digits + 1 sign bit), so the injection MLP always reads the same
        structure regardless of operation.

        Input:  [B, S, 5K+1] — all operations concatenated
        Output: [B, S, K+1]  — selected operation's digits + sign bit
        """
        # Skip if token tables haven't been built yet (unit tests with synthetic IDs)
        if self.extract.op_token_to_result_idx.size(0) <= 1:
            # Return first K+1 dims as fallback for unit tests
            return results[..., : self.num_digits + 1]

        B, S, _ = results.shape
        K = self.num_digits
        device = results.device

        # Find operator position and look up its result index (0-4)
        op_pos = self.extract._find_operator_positions(input_ids)  # [B]
        op_tokens = input_ids.gather(1, op_pos.unsqueeze(1)).squeeze(1)  # [B]
        op_clamped = op_tokens.clamp(0, self.extract.op_token_to_result_idx.size(0) - 1)
        op_indices = self.extract.op_token_to_result_idx[op_clamped]  # [B], 0-4 or -1

        # Pack all 5 operations into [B, S, 5, K+1] with one allocation.
        # Sign bit at position K is 0 (from zeros) for unsigned ops.
        K1 = K + 1
        stacked = torch.zeros(B, S, 5, K1, device=device, dtype=results.dtype)
        stacked[..., 0, :K] = results[..., 0:K]                   # add digits
        stacked[..., 1, :]  = results[..., K : 2 * K + 1]         # sub digits + sign
        stacked[..., 2, :K] = results[..., 2 * K + 1 : 3 * K + 1] # mul digits
        stacked[..., 3, :K] = results[..., 3 * K + 1 : 4 * K + 1] # exp digits
        stacked[..., 4, :K] = results[..., 4 * K + 1 : 5 * K + 1] # div digits

        # Gather the selected operation: index dim -2 with op_indices
        idx = op_indices.clamp(0, 4).view(B, 1, 1, 1).expand(B, S, 1, K1)
        selected = stacked.gather(-2, idx).squeeze(-2)  # [B, S, K+1]

        # Zero out if no valid operator was found
        valid = (op_indices >= 0).float().view(B, 1, 1)
        return selected * valid

    def _reorder_to_msb_first(self, results: Tensor) -> Tensor:
        """Reorder digit slots from LSB-first to MSB-first (left-aligned).

        CRT decode produces LSB-first: d[0]=ones, d[1]=tens, d[2]=hundreds.
        Text output is MSB-first: "579" outputs '5' then '7' then '9'.

        With LSB-first, the model at position P must compute:
            output = d[num_significant - 1 - P]
        — a conditional reverse-index that depends on answer length.

        With MSB-first, the model at position P simply reads d[P].
        Sentinel value -1 marks positions past the last significant digit,
        teaching the model when to emit a termination token.

        All operations are standard PyTorch tensor ops — no CPU, no loops.

        Args:
            results: [B, S, K+1] — K digits (LSB-first) + sign bit

        Returns:
            [B, S, K+1] — K digits (MSB-first, -1 padding) + sign bit
        """
        K = self.num_digits
        digits = results[..., :K]  # [B, S, K]
        sign = results[..., K:]    # [B, S, 1]
        device = digits.device

        # Index tensor [0, 1, ..., K-1] broadcast to all elements
        indices = torch.arange(K, device=device, dtype=torch.long)  # [K]

        # Find num_significant_digits per element:
        # highest non-zero index + 1, or 1 for all-zero (represents "0")
        nonzero = digits != 0                                       # [B, S, K]
        # Where nonzero, take the digit index; else -1
        highest = torch.where(
            nonzero, indices, torch.tensor(-1, device=device),
        ).amax(dim=-1)                                              # [B, S]
        num_sig = torch.where(
            nonzero.any(dim=-1), highest + 1, torch.ones_like(highest),
        )                                                           # [B, S]

        # For output position p, source from LSB index (num_sig - 1 - p)
        num_sig_k = num_sig.unsqueeze(-1)                           # [B, S, 1]
        source = (num_sig_k - 1 - indices).clamp(0, K - 1)         # [B, S, K]
        valid = indices < num_sig_k                                 # [B, S, K]

        msb_digits = digits.gather(-1, source)                      # [B, S, K]
        msb_digits = torch.where(valid, msb_digits, torch.full_like(msb_digits, -1.0))

        return torch.cat([msb_digits, sign], dim=-1)                # [B, S, K+1]

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
        # Generation cache: reuse cached results on step 2+
        if self._generation_mode and self._cached_results is not None:
            return self._forward_generation_cached(h)

        # Stage 1: Deterministic extraction via token lookup
        d_a, d_b, _, _ = self.extract(h, input_ids, attention_mask)

        # Stage 2: Encode to RNS circles
        a_circle = self.encode(d_a)          # [B, S, m, 2]
        b_circle = self.encode(d_b)          # [B, S, m, 2]
        b_exp_circle = self.encode.encode_exponent(d_b)  # [B, S, m, 2]

        # Stage 3: Compute + CRT decode to digit vectors
        all_results = self._decode_to_digits(a_circle, b_circle, b_exp_circle)  # [B, S, 5K+1]

        # Select the result matching the detected operator and relocate
        # to fixed positions [0:K+1], so injection always reads the same structure.
        results = self._select_operation_result(all_results, input_ids)  # [B, S, K+1]

        # Reorder digits from LSB-first (CRT output) to MSB-first (text order).
        # This lets the model at answer position P simply read d[P] = the P-th
        # most-significant digit, instead of learning a length-dependent
        # conditional reverse-index.  Sentinel -1 marks where to stop.
        results = self._reorder_to_msb_first(results)  # [B, S, K+1], now MSB-first

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

        # Apply mask to results
        masked_results = results * inject_mask.unsqueeze(2)  # [B, S, D]

        # Add position-aware answer offset embeddings if enabled
        if self.answer_pos_embed is not None:
            offset = (positions - eq_pos.unsqueeze(1)).clamp(0, 15)  # [B, S]
            offset = (offset * inject_mask).long()  # zero out non-answer positions
            pos_emb = self.answer_pos_embed(offset)  # [B, S, pos_dim]
            pos_emb = pos_emb * inject_mask.unsqueeze(2)
            masked_results = torch.cat([masked_results, pos_emb], dim=-1)

        # Stage 4: Inject decoded digits into hidden state
        h_prime = self.inject(masked_results, h)

        # Zero out entire injection delta for non-arithmetic sequences
        # (prevents MLP bias leakage when results are all zeros)
        seq_gate = has_eq.float().view(B, 1, 1)
        h_prime = h + (h_prime - h) * seq_gate

        # Cache results for generation mode (step 2+ will reuse)
        if self._generation_mode and self._cached_results is None:
            # Cache the unmasked digit results (position-independent, same at all positions)
            # Take the first position's results since they're broadcast-identical
            self._cached_results = results[:, 0, :].clone()
            self._cached_has_eq = has_eq.clone()
            # Next generation step will be at offset 1 from '='
            self._generation_offset = 1

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
