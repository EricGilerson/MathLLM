"""Stage 1: Deterministic Operand Extraction via Token Lookup.

Extraction is fully deterministic — no learned parameters.

Supports two tokenization styles:
- **Single-token numbers** (GPT-2): "300" is one token → direct lookup
- **Per-digit numbers** (SmolLM2, etc.): "300" is ['3','0','0'] → collect & assemble

The extractor:
1. Builds a frozen lookup table mapping token IDs to digit values.
2. Finds the operator token (+, -, *, /, ^) by scanning input_ids.
3. Collects digit tokens before the operator (operand A) and after (operand B).
4. Assembles digit vectors in LSB-first format.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class OperandExtractor(nn.Module):
    """Deterministic operand extraction via token ID lookup.

    No learned parameters. Scans input_ids for operator tokens,
    collects adjacent digit tokens, and assembles digit vectors.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_digits: int = 10,
        num_classes: int = 10,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
        use_attention: bool = False,
        attn_rank: int = 32,
    ):
        super().__init__()
        self.num_digits = num_digits

        # These buffers are properly sized by build_token_digits_table()
        # Placeholder until then:
        self.register_buffer(
            "token_digit_value", torch.full((1,), -1, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "is_operator", torch.zeros(1, dtype=torch.bool),
            persistent=True,
        )
        # Whether this tokenizer uses per-digit tokenization
        self._per_digit = False

    def build_token_digits_table(self, tokenizer) -> None:
        """Build frozen lookup tables from the tokenizer.

        Creates:
        - token_digit_value[V]: single-digit value (0-9) for digit tokens, -1 otherwise
        - is_operator[V]: True for operator tokens
        - _per_digit: True if numbers are tokenized as individual digits

        Also works for tokenizers where "300" is a single token (GPT-2 style).
        """
        vocab_size = tokenizer.vocab_size

        # Map each token to its digit value (-1 = not a digit)
        digit_val = torch.full((vocab_size,), -1, dtype=torch.long)
        # Track which tokens are multi-digit numbers (for GPT-2 style)
        multi_digit_table = {}

        for token_id in range(vocab_size):
            text = tokenizer.decode([token_id]).strip()
            if not text.isascii():
                continue
            if text.isdigit():
                val = int(text)
                if val <= 9:
                    digit_val[token_id] = val
                else:
                    # Multi-digit number token (GPT-2 style)
                    multi_digit_table[token_id] = val

        # Detect tokenization style: check if "300" is one token or multiple
        test_ids = tokenizer.encode("300", add_special_tokens=False)
        self._per_digit = len(test_ids) > 1

        # For single-token number tokenizers, also store full number → digit vectors
        if not self._per_digit and multi_digit_table:
            # Build a [V, K] table for direct lookup (GPT-2 style)
            full_table = torch.zeros(vocab_size, self.num_digits)
            # Single digits
            for tid in range(vocab_size):
                if digit_val[tid] >= 0:
                    full_table[tid, 0] = digit_val[tid].item()
            # Multi-digit numbers
            for tid, val in multi_digit_table.items():
                for d in range(self.num_digits):
                    full_table[tid, d] = val % 10
                    val //= 10
            self.register_buffer("token_digits_full", full_table, persistent=True)

        # Build operator mask
        is_op = torch.zeros(vocab_size, dtype=torch.bool)
        for op_char in ['+', '-', '*', '/', '^']:
            op_ids = tokenizer.encode(op_char, add_special_tokens=False)
            if len(op_ids) == 1:
                is_op[op_ids[0]] = True
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "Operator '%s' encodes to %d tokens (expected 1); "
                    "extraction may not detect this operator.",
                    op_char, len(op_ids),
                )

        self.token_digit_value = digit_val
        self.is_operator = is_op

    def _find_operator_positions(self, input_ids: Tensor) -> Tensor:
        """Find the position of the first operator token in each sequence."""
        ids_clamped = input_ids.clamp(0, self.is_operator.size(0) - 1)
        is_op = self.is_operator[ids_clamped]  # [B, S] bool
        op_pos = is_op.long().argmax(dim=1)  # [B]
        return op_pos

    def _extract_single_token(
        self, input_ids: Tensor, op_pos: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Extract operands for single-token number tokenizers (GPT-2 style).

        Operand A is the token at op_pos - 1, operand B at op_pos + 1.
        """
        B, S = input_ids.shape
        a_pos = (op_pos - 1).clamp(min=0)
        b_pos = (op_pos + 1).clamp(max=S - 1)

        a_ids = input_ids.gather(1, a_pos.unsqueeze(1)).squeeze(1)
        b_ids = input_ids.gather(1, b_pos.unsqueeze(1)).squeeze(1)

        a_clamped = a_ids.clamp(0, self.token_digits_full.size(0) - 1)
        b_clamped = b_ids.clamp(0, self.token_digits_full.size(0) - 1)

        d_a = self.token_digits_full[a_clamped]  # [B, K]
        d_b = self.token_digits_full[b_clamped]  # [B, K]
        return d_a, d_b

    def _extract_per_digit(
        self, input_ids: Tensor, op_pos: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Extract operands for per-digit tokenizers (SmolLM2 style).

        Collects consecutive digit tokens before the operator (A) and
        between the operator and '=' (B), then assembles LSB-first.
        """
        B, S = input_ids.shape
        K = self.num_digits
        device = input_ids.device

        # Get digit values for each token: [B, S], -1 for non-digits
        ids_clamped = input_ids.clamp(0, self.token_digit_value.size(0) - 1)
        dv = self.token_digit_value[ids_clamped]  # [B, S]
        digit_mask = dv >= 0
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        digit_offsets = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)

        # Operand A is the contiguous run of digit tokens immediately before the operator.
        invalid_before = (~digit_mask) & (positions < op_pos.unsqueeze(1))
        last_invalid_before = torch.where(
            invalid_before,
            positions + 1,
            torch.zeros_like(positions),
        ).amax(dim=1)
        a_start = last_invalid_before
        a_positions = op_pos.unsqueeze(1) - 1 - digit_offsets
        a_in_range = a_positions >= a_start.unsqueeze(1)
        a_indices = a_positions.clamp(0, S - 1)
        d_a = torch.where(
            a_in_range,
            dv.gather(1, a_indices),
            torch.zeros(B, K, device=device, dtype=dv.dtype),
        ).to(dtype=torch.float32)

        # Operand B is the contiguous run of digit tokens immediately after the operator.
        invalid_after = (~digit_mask) & (positions > op_pos.unsqueeze(1))
        first_invalid_after = torch.where(
            invalid_after,
            positions,
            torch.full_like(positions, S),
        ).amin(dim=1)
        b_end = first_invalid_after - 1
        b_positions = b_end.unsqueeze(1) - digit_offsets
        b_in_range = b_positions > op_pos.unsqueeze(1)
        b_indices = b_positions.clamp(0, S - 1)
        d_b = torch.where(
            b_in_range,
            dv.gather(1, b_indices),
            torch.zeros(B, K, device=device, dtype=dv.dtype),
        ).to(dtype=torch.float32)

        return d_a, d_b

    def forward(
        self,
        h: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Deterministic operand extraction.

        Args:
            h: Hidden state [batch, seq_len, hidden_dim] (unused, kept for API)
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] (unused, kept for API)

        Returns:
            d_a: Digit vector for operand A [batch, seq_len, K]
            d_b: Digit vector for operand B [batch, seq_len, K]
            d_a: Same (no soft/hard distinction in deterministic mode)
            d_b: Same
        """
        B, S = input_ids.shape
        K = self.num_digits

        op_pos = self._find_operator_positions(input_ids)

        if self._per_digit:
            d_a_flat, d_b_flat = self._extract_per_digit(input_ids, op_pos)
        else:
            d_a_flat, d_b_flat = self._extract_single_token(input_ids, op_pos)

        # Broadcast to all sequence positions: [B, K] -> [B, S, K]
        d_a = d_a_flat.unsqueeze(1).expand(B, S, K)
        d_b = d_b_flat.unsqueeze(1).expand(B, S, K)

        return d_a, d_b, d_a, d_b
