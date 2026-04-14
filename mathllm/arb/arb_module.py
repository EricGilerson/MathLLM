"""Arithmetic Residual Block: compute-once, inject-many architecture.

The ARB pipeline has 4 stages:
  1. Extract (deterministic) — read digit vectors from token IDs
  2. Encode (frozen) — map to RNS circle representation
  3. Compute (frozen) — execute +, -, *, exp, / in parallel
  4. Inject (learned) — project results back into hidden state

Stages 1-3 produce identical output regardless of transformer depth, so they
run once per forward pass in ``ARBComputeCore``.  Stage 4 runs at each
configured layer position via per-layer ``ARBInjector`` instances.

Smart activation: injection only fires at the **last** ``=`` in the sequence
and its contiguous digit suffix — so already-answered equations in context
(``3+5=8, now compute 7*9=``) don't get spurious injection.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn
from torch import Tensor

from mathllm.arb.stage1_extract import OperandExtractor
from mathllm.arb.stage2_encode import RNSCircleEncoder
from mathllm.arb.stage3_compute import ArithmeticCompute
from mathllm.arb.stage4_inject import ResultInjector

# Legacy default for GPT-2 ('=' is token 28)
_DEFAULT_EQ_TOKEN_ID = 28

_MAX_ANSWER_TOKENS = 16  # enough for any answer length


@dataclass
class ComputeResult:
    """Output of ``ARBComputeCore.forward()``."""

    results: Tensor   # [B, S, K+1] MSB-first answer digits + sign
    d_a: Tensor       # [B, S, K]   operand A digit vector
    d_b: Tensor       # [B, S, K]   operand B digit vector
    has_eq: Tensor    # [B]         whether each sequence contains '='
    eq_pos: Tensor    # [B]         position of last '=' (or S if absent)


class ARBComputeCore(nn.Module):
    """Stages 1-3: deterministic extraction, RNS encoding, and arithmetic.

    Runs once per forward pass.  Does not touch the hidden state — produces
    a digit-level answer tensor that ``ARBInjector`` instances consume.
    """

    def __init__(
        self,
        hidden_dim: int,
        primes: tuple[int, ...] = (7, 11, 13, 17, 19, 23, 29, 31, 37),
        num_digits: int = 10,
        num_results: int = 5,
        softmax_temperature: float = 1000.0,
        num_classes: int = 10,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
        use_attention: bool = False,
        attn_rank: int = 32,
        eq_token_id: int = _DEFAULT_EQ_TOKEN_ID,
    ):
        super().__init__()
        self.num_digits = num_digits
        self.num_results = num_results
        self.eq_token_id = eq_token_id

        # Stage 1: Deterministic extraction
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

        # Freeze stages 2 and 3
        for param in self.encode.parameters():
            param.requires_grad = False
        for param in self.compute.parameters():
            param.requires_grad = False

        # Generation cache state
        self._generation_mode = False
        self._cached_results: Tensor | None = None
        self._cached_has_eq: Tensor | None = None
        self._generation_offset = 0

    # ------------------------------------------------------------------
    # Generation cache mode
    # ------------------------------------------------------------------

    def enter_generation_mode(self) -> None:
        """Enable generation cache.  First forward caches; subsequent reuse."""
        self._generation_mode = True
        self._cached_results = None
        self._cached_has_eq = None
        self._generation_offset = 0

    def exit_generation_mode(self) -> None:
        """Disable generation cache and clear state."""
        self._generation_mode = False
        self._cached_results = None
        self._cached_has_eq = None
        self._generation_offset = 0

    @property
    def generation_offset(self) -> int:
        return self._generation_offset

    def advance_generation_offset(self) -> None:
        self._generation_offset += 1

    # ------------------------------------------------------------------
    # Device warm-up
    # ------------------------------------------------------------------

    def prepare_for_device(self, device: torch.device | str) -> None:
        """Warm runtime-only frozen buffers after the module is moved."""
        self.compute.prepare_for_device(device)

    # ------------------------------------------------------------------
    # Internal compute methods
    # ------------------------------------------------------------------

    def _decode_to_digits(
        self,
        a_circle: Tensor,
        b_circle: Tensor,
        b_exp_circle: Tensor,
    ) -> Tensor:
        """Compute all operations and decode results to digit vectors.

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
            add_n = self.compute.crt_reconstruct(add_c)
            sub_n = self.compute.crt_reconstruct_signed(sub_c)
            mul_n = self.compute.crt_reconstruct(mul_c)
            exp_n = self.compute.crt_reconstruct(exp_c)
            div_n = self.compute.crt_reconstruct(div_c)

            add_digits = self.compute.integer_to_digits(add_n)
            sub_digits = self.compute.integer_to_digits_with_sign(sub_n)
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

        Input:  [B, S, 5K+1] — all operations concatenated
        Output: [B, S, K+1]  — selected operation's digits + sign bit
        """
        # Skip if token tables haven't been built yet (unit tests with synthetic IDs)
        if self.extract.op_token_to_result_idx.size(0) <= 1:
            return results[..., : self.num_digits + 1]

        B, S, _ = results.shape
        K = self.num_digits
        device = results.device

        op_pos = self.extract._find_operator_positions(input_ids)
        op_tokens = input_ids.gather(1, op_pos.unsqueeze(1)).squeeze(1)
        op_clamped = op_tokens.clamp(0, self.extract.op_token_to_result_idx.size(0) - 1)
        op_indices = self.extract.op_token_to_result_idx[op_clamped]

        K1 = K + 1
        stacked = torch.zeros(B, S, 5, K1, device=device, dtype=results.dtype)
        stacked[..., 0, :K] = results[..., 0:K]
        stacked[..., 1, :]  = results[..., K : 2 * K + 1]
        stacked[..., 2, :K] = results[..., 2 * K + 1 : 3 * K + 1]
        stacked[..., 3, :K] = results[..., 3 * K + 1 : 4 * K + 1]
        stacked[..., 4, :K] = results[..., 4 * K + 1 : 5 * K + 1]

        idx = op_indices.clamp(0, 4).view(B, 1, 1, 1).expand(B, S, 1, K1)
        selected = stacked.gather(-2, idx).squeeze(-2)

        valid = (op_indices >= 0).float().view(B, 1, 1)
        return selected * valid

    def _reorder_to_msb_first(self, results: Tensor) -> Tensor:
        """Reorder digit slots from LSB-first to MSB-first (left-aligned).

        Sentinel value -1 marks positions past the last significant digit.
        """
        K = self.num_digits
        digits = results[..., :K]
        sign = results[..., K:]
        device = digits.device

        indices = torch.arange(K, device=device, dtype=torch.long)
        nonzero = digits != 0
        highest = torch.where(
            nonzero, indices, torch.tensor(-1, device=device),
        ).amax(dim=-1)
        num_sig = torch.where(
            nonzero.any(dim=-1), highest + 1, torch.ones_like(highest),
        )

        num_sig_k = num_sig.unsqueeze(-1)
        source = (num_sig_k - 1 - indices).clamp(0, K - 1)
        valid = indices < num_sig_k

        msb_digits = digits.gather(-1, source)
        msb_digits = torch.where(valid, msb_digits, torch.full_like(msb_digits, -1.0))

        return torch.cat([msb_digits, sign], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> ComputeResult:
        """Run stages 1-3 and return digit-level results.

        Args:
            input_ids: [B, S] token IDs
            attention_mask: [B, S] (1 = real, 0 = padding)

        Returns:
            ComputeResult with answer digits, operand digits, and eq metadata.
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Find '=' position (used by both compute core and injectors)
        eq_present = (input_ids == self.eq_token_id)
        has_eq = eq_present.any(dim=1)
        # Use the LAST '=' in the sequence (not argmax which gives the first)
        # Flip, argmax on flipped gives first-from-end, then convert back
        eq_pos = torch.where(
            has_eq,
            S - 1 - eq_present.long().flip(dims=[1]).argmax(dim=1),
            torch.full((B,), S, device=device),
        )

        # Generation cache: return cached results on step 2+
        if self._generation_mode and self._cached_results is not None:
            K = self.num_digits
            d_a = torch.zeros(B, 1, K, device=device)
            d_b = torch.zeros(B, 1, K, device=device)
            return ComputeResult(
                results=self._cached_results.unsqueeze(1),
                d_a=d_a,
                d_b=d_b,
                has_eq=self._cached_has_eq if self._cached_has_eq is not None else has_eq,
                eq_pos=eq_pos,
            )

        # Stage 1: Deterministic extraction via token lookup
        d_a, d_b, _, _ = self.extract(None, input_ids, attention_mask)

        # Stage 2: Encode to RNS circles
        a_circle = self.encode(d_a)
        b_circle = self.encode(d_b)
        b_exp_circle = self.encode.encode_exponent(d_b)

        # Stage 3: Compute + CRT decode to digit vectors
        all_results = self._decode_to_digits(a_circle, b_circle, b_exp_circle)

        # Select the result matching the detected operator
        results = self._select_operation_result(all_results, input_ids)

        # Reorder to MSB-first for left-to-right generation
        results = self._reorder_to_msb_first(results)

        # Cache for generation mode (step 2+ will reuse)
        if self._generation_mode and self._cached_results is None:
            self._cached_results = results[:, 0, :].clone()
            self._cached_has_eq = has_eq.clone()
            self._generation_offset = 1

        return ComputeResult(
            results=results,
            d_a=d_a,
            d_b=d_b,
            has_eq=has_eq,
            eq_pos=eq_pos,
        )


class DigitSelector(nn.Module):
    """Position-aware digit selection: hard index + soft attention context.

    Replaces the naive concatenation of the full digit vector with a cleaner
    signal for the projection MLP.  At answer position k:

    - **Hard index**: directly selects ``digits[k]`` (no learning needed).
    - **Soft attention**: a tiny single-head attention over all K digit slots,
      queried by the position embedding.  Provides magnitude / neighbour
      context without forcing the MLP to learn implicit selection.
    """

    def __init__(
        self,
        num_digits: int,
        pos_dim: int,
        attn_dim: int = 8,
        hard_select: bool = True,
    ):
        super().__init__()
        self.num_digits = num_digits
        self.attn_dim = attn_dim
        self.hard_select = hard_select

        if attn_dim > 0:
            self.W_q = nn.Linear(pos_dim, attn_dim, bias=False)
            self.W_k = nn.Linear(1, attn_dim, bias=False)
            self.W_v = nn.Linear(1, attn_dim, bias=False)
            self.slot_embed = nn.Parameter(
                torch.randn(num_digits, attn_dim) * 0.02
            )
            self.attn_scale = math.sqrt(attn_dim)

    @property
    def output_dim(self) -> int:
        """Total output dimension *excluding* ``pos_emb`` (caller appends)."""
        d = 0
        if self.hard_select:
            d += 1           # hard-selected digit value
        if self.attn_dim > 0:
            d += self.attn_dim  # soft attention context vector
        d += 1               # sign bit
        return d

    def forward(
        self,
        digits: Tensor,
        sign: Tensor,
        pos_emb: Tensor,
        offset: Tensor,
    ) -> Tensor:
        """Select digit information for each answer position.

        Args:
            digits:  [B, S, K]       — MSB-first digit values (-1 = sentinel)
            sign:    [B, S, 1]       — sign bit
            pos_emb: [B, S, pos_dim] — position embeddings
            offset:  [B, S]          — integer offsets from ``=``

        Returns:
            [B, S, output_dim + pos_dim] — combined features for the MLP.
        """
        B, S, K = digits.shape
        parts: list[Tensor] = []

        # --- Hard selection: pick the single relevant digit ---
        if self.hard_select:
            k_idx = offset.clamp(0, K - 1).unsqueeze(-1)   # [B, S, 1]
            d_hard = digits.gather(-1, k_idx)                # [B, S, 1]
            d_hard = torch.where(d_hard < 0, torch.zeros_like(d_hard), d_hard)
            parts.append(d_hard)

        # --- Soft attention context over all digit slots ---
        if self.attn_dim > 0:
            q = self.W_q(pos_emb)                            # [B, S, D_a]

            d_exp = digits.unsqueeze(-1)                     # [B, S, K, 1]
            keys = self.W_k(d_exp) + self.slot_embed         # [B, S, K, D_a]
            vals = self.W_v(d_exp) + self.slot_embed         # [B, S, K, D_a]

            # Scaled dot-product:  [B, S, 1, D_a] @ [B, S, D_a, K] → [B, S, K]
            scores = torch.matmul(
                q.unsqueeze(-2), keys.transpose(-1, -2),
            ).squeeze(-2) / self.attn_scale

            # Mask sentinel positions
            sentinel_mask = digits < 0                       # [B, S, K]
            scores = scores.masked_fill(sentinel_mask, -1e9)

            attn_w = torch.softmax(scores, dim=-1)           # [B, S, K]
            ctx = (attn_w.unsqueeze(-1) * vals).sum(dim=-2)  # [B, S, D_a]
            parts.append(ctx)

        parts.append(sign)
        parts.append(pos_emb)
        return torch.cat(parts, dim=-1)


class ARBInjector(nn.Module):
    """Stage 4: per-layer injection with smart activation masking.

    Receives the shared answer tensor from ``ARBComputeCore`` and projects it
    into the hidden state via a learned gated residual.  The smart activation
    mask ensures injection only fires at the last ``=`` and its contiguous
    digit suffix — already-answered equations in context are left alone.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_digits: int = 10,
        dropout: float = 0.1,
        injector_init_std: float = 1e-3,
        gate_init_logit: float = -2.0,
        injection_pos_dim: int = 0,
        injection_mlp_hidden: int = 0,
        injection_attn_dim: int = 0,
        injection_hard_select: bool = False,
    ):
        super().__init__()
        self.num_digits = num_digits
        self.injection_pos_dim = injection_pos_dim
        self._use_digit_selector = (
            injection_pos_dim > 0
            and (injection_hard_select or injection_attn_dim > 0)
        )

        digit_result_dim = num_digits + 1  # K digits + sign bit

        if injection_pos_dim > 0:
            self.answer_pos_embed = nn.Embedding(_MAX_ANSWER_TOKENS, injection_pos_dim)
        else:
            self.answer_pos_embed = None

        if self._use_digit_selector:
            self.digit_selector = DigitSelector(
                num_digits=num_digits,
                pos_dim=injection_pos_dim,
                attn_dim=injection_attn_dim,
                hard_select=injection_hard_select,
            )
            total_result_dim = self.digit_selector.output_dim + injection_pos_dim
        elif injection_pos_dim > 0:
            self.digit_selector = None
            total_result_dim = digit_result_dim + injection_pos_dim
        else:
            self.digit_selector = None
            total_result_dim = digit_result_dim

        self.inject = ResultInjector(
            hidden_dim,
            total_result_dim,
            dropout=dropout,
            init_std=injector_init_std,
            gate_init_logit=gate_init_logit,
            mlp_hidden=injection_mlp_hidden,
        )

        # Populated by build_token_digit_tables via the model
        self.register_buffer(
            "token_digit_value",
            torch.full((1,), -1, dtype=torch.long),
            persistent=False,
        )

    def set_token_digit_value(self, buf: Tensor) -> None:
        """Share the digit lookup buffer from the compute core."""
        self.token_digit_value = buf

    def _build_smart_inject_mask(
        self,
        input_ids: Tensor,
        eq_pos: Tensor,
        has_eq: Tensor,
    ) -> Tensor:
        """Build injection mask: activate only at the answer zone.

        The answer zone is the last '=' and the contiguous block of digit
        tokens immediately following it.  If no digits follow '=' (generation
        prompt), the mask extends from '=' to end of sequence.

        Returns:
            [B, S] float mask (1.0 = inject, 0.0 = skip)
        """
        B, S = input_ids.shape
        device = input_ids.device

        positions = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
        at_eq = positions == eq_pos.unsqueeze(1)                  # [B, S]
        after_eq = positions > eq_pos.unsqueeze(1)                # [B, S]

        # Determine which positions are digit tokens
        ids_clamped = input_ids.clamp(0, self.token_digit_value.size(0) - 1)
        is_digit = self.token_digit_value[ids_clamped] >= 0       # [B, S]

        # Find the first non-digit position after '='
        non_digit_after_eq = after_eq & ~is_digit
        # For positions that aren't non-digit-after-eq, use S (so they
        # don't affect the min).  Then take the minimum per sequence.
        first_non_digit = torch.where(
            non_digit_after_eq,
            positions.expand(B, -1),
            torch.full((B, S), S, device=device, dtype=torch.long),
        ).amin(dim=1)  # [B]

        # Answer zone: at '=' OR (after '=' AND before first non-digit)
        in_answer_zone = at_eq | (after_eq & (positions < first_non_digit.unsqueeze(1)))

        # Zero out for sequences without '='
        return in_answer_zone.float() * has_eq.float().unsqueeze(1)

    def forward(
        self,
        h: Tensor,
        results: Tensor,
        has_eq: Tensor,
        eq_pos: Tensor,
        input_ids: Tensor,
        generation_offset: int | None = None,
    ) -> Tensor:
        """Inject the answer tensor into the hidden state.

        Args:
            h: [B, S, hidden_dim] — hidden state from transformer layer
            results: [B, S, K+1] — answer digits from compute core
            has_eq: [B] — whether each sequence contains '='
            eq_pos: [B] — position of last '='
            input_ids: [B, S] — token IDs (for smart mask digit detection)
            generation_offset: if not None, we are in generation step 2+;
                use this as the answer position offset.

        Returns:
            h': [B, S, hidden_dim]
        """
        B, S = h.shape[:2]
        device = h.device

        K = self.num_digits

        if generation_offset is not None:
            # Generation cache path (step 2+): results is [B, 1, K+1]
            if self._use_digit_selector:
                digits = results[..., :K]                    # [B, 1, K]
                sign = results[..., K:]                      # [B, 1, 1]
                offset_val = min(generation_offset, _MAX_ANSWER_TOKENS - 1)
                offset_tensor = torch.full(
                    (B, 1), offset_val, dtype=torch.long, device=device
                )
                pos_emb = self.answer_pos_embed(offset_tensor)
                selected = self.digit_selector(digits, sign, pos_emb, offset_tensor)
                h_prime = self.inject(selected, h)
            else:
                masked_results = results
                if self.answer_pos_embed is not None:
                    offset = min(generation_offset, _MAX_ANSWER_TOKENS - 1)
                    offset_tensor = torch.full(
                        (B, 1), offset, dtype=torch.long, device=device
                    )
                    pos_emb = self.answer_pos_embed(offset_tensor)
                    masked_results = torch.cat([masked_results, pos_emb], dim=-1)
                h_prime = self.inject(masked_results, h)

            # Sequence-level gate
            seq_gate = has_eq.float().view(B, 1, 1)
            return h + (h_prime - h) * seq_gate

        # Normal path: build smart injection mask
        inject_mask = self._build_smart_inject_mask(input_ids, eq_pos, has_eq)
        masked_results = results * inject_mask.unsqueeze(2)

        if self._use_digit_selector:
            digits = masked_results[..., :K]                 # [B, S, K]
            sign = masked_results[..., K:]                   # [B, S, 1]
            positions = torch.arange(S, device=device).unsqueeze(0)
            offset = (positions - eq_pos.unsqueeze(1)).clamp(0, _MAX_ANSWER_TOKENS - 1)
            offset = (offset * inject_mask).long()           # [B, S]
            pos_emb = self.answer_pos_embed(offset)
            pos_emb = pos_emb * inject_mask.unsqueeze(2)
            selected = self.digit_selector(digits, sign, pos_emb, offset)
            selected = selected * inject_mask.unsqueeze(2)
            h_prime = self.inject(selected, h)
        else:
            # Legacy path: concatenate full digit vector + pos embedding
            if self.answer_pos_embed is not None:
                positions = torch.arange(S, device=device).unsqueeze(0)
                offset = (positions - eq_pos.unsqueeze(1)).clamp(0, _MAX_ANSWER_TOKENS - 1)
                offset = (offset * inject_mask).long()
                pos_emb = self.answer_pos_embed(offset)
                pos_emb = pos_emb * inject_mask.unsqueeze(2)
                masked_results = torch.cat([masked_results, pos_emb], dim=-1)
            h_prime = self.inject(masked_results, h)

        # Sequence-level gate (prevents MLP bias leakage on non-arithmetic)
        seq_gate = has_eq.float().view(B, 1, 1)
        return h + (h_prime - h) * seq_gate


# ======================================================================
# Deprecated: monolithic ARB (kept for backward compatibility in tests)
# ======================================================================

class ArithmeticResidualBlock(nn.Module):
    """Complete Arithmetic Residual Block (deprecated).

    Prefer ``ARBComputeCore`` + ``ARBInjector`` for new code.
    This wrapper is kept so that standalone unit tests still work.
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
        injection_attn_dim: int = 0,
        injection_hard_select: bool = False,
    ):
        super().__init__()
        self.num_digits = num_digits
        self.eq_token_id = eq_token_id

        self.core = ARBComputeCore(
            hidden_dim=hidden_dim,
            primes=primes,
            num_digits=num_digits,
            num_results=num_results,
            softmax_temperature=softmax_temperature,
            num_classes=num_classes,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            use_attention=use_attention,
            attn_rank=attn_rank,
            eq_token_id=eq_token_id,
        )

        self.injector = ARBInjector(
            hidden_dim=hidden_dim,
            num_digits=num_digits,
            dropout=dropout,
            injector_init_std=injector_init_std,
            gate_init_logit=gate_init_logit,
            injection_pos_dim=injection_pos_dim,
            injection_mlp_hidden=injection_mlp_hidden,
            injection_attn_dim=injection_attn_dim,
            injection_hard_select=injection_hard_select,
        )

        # Expose sub-modules for direct access (backward compat)
        self.extract = self.core.extract
        self.encode = self.core.encode
        self.compute = self.core.compute
        self.inject = self.injector.inject
        self.answer_pos_embed = self.injector.answer_pos_embed

    # Delegate internal methods for tests that call them directly
    def _select_operation_result(self, results: Tensor, input_ids: Tensor) -> Tensor:
        return self.core._select_operation_result(results, input_ids)

    def _reorder_to_msb_first(self, results: Tensor) -> Tensor:
        return self.core._reorder_to_msb_first(results)

    def _decode_to_digits(self, a_circle: Tensor, b_circle: Tensor, b_exp_circle: Tensor) -> Tensor:
        return self.core._decode_to_digits(a_circle, b_circle, b_exp_circle)

    def enter_generation_mode(self) -> None:
        self.core.enter_generation_mode()

    def exit_generation_mode(self) -> None:
        self.core.exit_generation_mode()

    def prepare_for_device(self, device: torch.device | str) -> None:
        self.core.prepare_for_device(device)

    def forward(
        self,
        h: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run full ARB (compute + inject).

        Returns:
            (h', d_a, d_b, answer)
        """
        cr = self.core(input_ids, attention_mask)

        gen_offset = None
        if self.core._generation_mode and self.core._cached_results is not None:
            # On the prompt pass, _cached_results is set DURING core.forward(),
            # so generation_offset should only be used on step 2+.
            # Detect step 2+ by checking if results is [B, 1, ...] (cached)
            if cr.results.shape[1] == 1 and h.shape[1] == 1:
                gen_offset = self.core.generation_offset
                self.core.advance_generation_offset()

        h_prime = self.injector(
            h, cr.results, cr.has_eq, cr.eq_pos, input_ids,
            generation_offset=gen_offset,
        )
        return h_prime, cr.d_a, cr.d_b, cr.results
