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

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class EquationDetection:
    """Token-level detection result for a supported direct equation."""

    has_valid_equation: Tensor  # [B] bool
    eq_pos: Tensor              # [B] selected valid '=' position, or S
    op_pos: Tensor              # [B] operator tied to eq_pos, or 0 when absent
    candidate_eq_count: Tensor  # [B] count of '=' tokens
    valid_eq_count: Tensor      # [B] count of syntactically valid equations


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
        self.register_buffer(
            "op_token_to_result_idx", torch.full((1,), -1, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "is_number_token", torch.zeros(1, dtype=torch.bool),
            # Derived from the tokenizer at load time, so old checkpoints must
            # not be required to contain it.
            persistent=False,
        )
        self.register_buffer(
            "is_whitespace_token", torch.zeros(1, dtype=torch.bool),
            # Derived from the tokenizer at load time, like is_number_token.
            persistent=False,
        )
        self.register_buffer(
            "is_equals_token", torch.zeros(1, dtype=torch.bool),
            # Derived from the tokenizer at load time, like is_number_token.
            persistent=False,
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
        is_number_token = torch.zeros(vocab_size, dtype=torch.bool)
        is_whitespace_token = torch.zeros(vocab_size, dtype=torch.bool)
        is_op = torch.zeros(vocab_size, dtype=torch.bool)
        op_result_idx = torch.full((vocab_size,), -1, dtype=torch.long)
        is_equals_token = torch.zeros(vocab_size, dtype=torch.bool)
        op_to_index = {'+': 0, '-': 1, '*': 2, '^': 3, '/': 4}
        # Track which tokens are multi-digit numbers (for GPT-2 style)
        multi_digit_table = {}

        for token_id in range(vocab_size):
            text = tokenizer.decode([token_id]).strip()
            raw_text = tokenizer.decode([token_id])
            if raw_text and raw_text.isspace():
                is_whitespace_token[token_id] = True
            if text in op_to_index:
                is_op[token_id] = True
                op_result_idx[token_id] = op_to_index[text]
            if text == "=":
                is_equals_token[token_id] = True
            if not text.isascii():
                continue
            if text.isdigit():
                val = int(text)
                if val <= 9:
                    digit_val[token_id] = val
                    is_number_token[token_id] = True
                else:
                    # Multi-digit number token (GPT-2 style)
                    multi_digit_table[token_id] = val
                    is_number_token[token_id] = True

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

        self.token_digit_value = digit_val
        self.is_number_token = is_number_token
        self.is_whitespace_token = is_whitespace_token
        self.is_equals_token = is_equals_token
        self.is_operator = is_op
        self.register_buffer("op_token_to_result_idx", op_result_idx, persistent=True)

    def _find_operator_positions(self, input_ids: Tensor) -> Tensor:
        """Find the position of the first operator token in each sequence."""
        ids_clamped = input_ids.clamp(0, self.is_operator.size(0) - 1)
        is_op = self.is_operator[ids_clamped]  # [B, S] bool
        op_pos = is_op.long().argmax(dim=1)  # [B]
        return op_pos

    def find_valid_equations(self, input_ids: Tensor, eq_token_id: int) -> EquationDetection:
        """Find supported direct ``A op B =`` expressions.

        The selected operator and equals sign always belong to the same local
        expression. This intentionally supports only the direct syntax used by
        the ARB benchmark; it does not attempt to parse natural-language or
        arbitrary programming-language expressions.
        """
        B, S = input_ids.shape
        device = input_ids.device
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        if self.is_equals_token.size(0) > 1:
            eq_ids = input_ids.clamp(0, self.is_equals_token.size(0) - 1)
            eq_mask = self.is_equals_token[eq_ids]
        else:
            eq_mask = input_ids == eq_token_id
        candidate_eq_count = eq_mask.sum(dim=1)
        next_is_eq = torch.zeros_like(eq_mask)
        if S > 1:
            next_is_eq[:, :-1] = eq_mask[:, 1:]

        ids_clamped = input_ids.clamp(0, self.is_operator.size(0) - 1)
        is_op = self.is_operator[ids_clamped]
        whitespace_ids = input_ids.clamp(0, self.is_whitespace_token.size(0) - 1)
        is_whitespace = self.is_whitespace_token[whitespace_ids]

        # Previous non-whitespace token for each position. This permits the
        # documented spaced form ``A op B =`` without accepting arbitrary text
        # between its components.
        non_whitespace_positions = torch.where(
            ~is_whitespace, positions, torch.full_like(positions, -1)
        )
        last_non_whitespace = non_whitespace_positions.cummax(dim=1).values
        previous_non_whitespace = torch.cat(
            [torch.full((B, 1), -1, device=device, dtype=torch.long), last_non_whitespace[:, :-1]],
            dim=1,
        )

        if self._per_digit:
            digit_ids = input_ids.clamp(0, self.token_digit_value.size(0) - 1)
            is_digit = self.token_digit_value[digit_ids] >= 0

            # Locate the last B digit before '='. The last non-digit,
            # non-whitespace token before that digit must be the operator.
            # Whitespace may separate components but never appears inside a
            # digit run.
            b_end = previous_non_whitespace.gather(1, positions)
            delimiter_positions = torch.where(
                ~is_digit & ~is_whitespace, positions, torch.full_like(positions, -1)
            )
            last_delimiter = delimiter_positions.cummax(dim=1).values
            op_pos = last_delimiter.gather(1, b_end.clamp(0, S - 1))
            a_pos = previous_non_whitespace.gather(
                1, op_pos.clamp(0, S - 1)
            )
            op_in_range = op_pos >= 1  # leaves room for at least one A digit
            op_indices = op_pos.clamp(0, S - 1)
            b_indices = b_end.clamp(0, S - 1)
            a_indices = a_pos.clamp(0, S - 1)
            valid_mask = (
                eq_mask
                & ~next_is_eq
                & (b_end >= 0)
                & op_in_range
                & is_digit.gather(1, b_indices)
                & is_op.gather(1, op_indices)
                & is_digit.gather(1, a_indices)
            )
        else:
            number_ids = input_ids.clamp(0, self.is_number_token.size(0) - 1)
            is_number = self.is_number_token[number_ids]
            b_indices = previous_non_whitespace
            op_indices = previous_non_whitespace.gather(
                1, b_indices.clamp(0, S - 1)
            )
            a_indices = previous_non_whitespace.gather(
                1, op_indices.clamp(0, S - 1)
            )
            op_pos = op_indices
            valid_mask = (
                eq_mask
                & ~next_is_eq
                & (b_indices >= 0)
                & (op_indices >= 0)
                & (a_indices >= 0)
                & is_number.gather(1, b_indices.clamp(0, S - 1))
                & is_op.gather(1, op_indices.clamp(0, S - 1))
                & is_number.gather(1, a_indices.clamp(0, S - 1))
            )

        has_valid_equation = valid_mask.any(dim=1)
        reversed_valid = valid_mask.long().flip(dims=[1]).argmax(dim=1)
        eq_pos = torch.where(
            has_valid_equation,
            S - 1 - reversed_valid,
            torch.full((B,), S, device=device, dtype=torch.long),
        )

        if self._per_digit:
            selected_op_pos = op_pos.gather(
                1, eq_pos.clamp(0, S - 1).unsqueeze(1)
            ).squeeze(1)
        else:
            selected_op_pos = eq_pos - 2
        selected_op_pos = torch.where(
            has_valid_equation,
            selected_op_pos,
            torch.zeros_like(selected_op_pos),
        )

        return EquationDetection(
            has_valid_equation=has_valid_equation,
            eq_pos=eq_pos,
            op_pos=selected_op_pos,
            candidate_eq_count=candidate_eq_count,
            valid_eq_count=valid_mask.sum(dim=1),
        )

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

        whitespace_ids = input_ids.clamp(0, self.is_whitespace_token.size(0) - 1)
        whitespace_mask = self.is_whitespace_token[whitespace_ids]

        # Arithmetic tokens may carry or be separated by whitespace. Locate the
        # nearest non-whitespace token on each side of the operator, then read
        # contiguous digit runs from those positions.
        non_whitespace_positions = torch.where(
            ~whitespace_mask, positions, torch.full_like(positions, -1)
        )
        last_non_whitespace = non_whitespace_positions.cummax(dim=1).values
        previous_non_whitespace = torch.cat(
            [torch.full((B, 1), -1, device=device, dtype=torch.long), last_non_whitespace[:, :-1]],
            dim=1,
        )
        next_non_whitespace_positions = torch.where(
            ~whitespace_mask, positions, torch.full_like(positions, S)
        )
        next_non_whitespace = next_non_whitespace_positions.flip(dims=[1]).cummin(dim=1).values.flip(dims=[1])
        next_non_whitespace_after = torch.cat(
            [next_non_whitespace[:, 1:], torch.full((B, 1), S, device=device, dtype=torch.long)],
            dim=1,
        )

        a_end = previous_non_whitespace.gather(1, op_pos.clamp(0, S - 1).unsqueeze(1)).squeeze(1)
        b_start = next_non_whitespace_after.gather(1, op_pos.clamp(0, S - 1).unsqueeze(1)).squeeze(1)

        # Operand A is the contiguous digit run ending before the operator.
        a_positions = a_end.unsqueeze(1) - digit_offsets
        a_in_range = a_positions >= 0
        a_indices = a_positions.clamp(0, S - 1)
        a_is_digit = digit_mask.gather(1, a_indices) & a_in_range
        a_contiguous = a_is_digit.long().cumprod(dim=1).bool()
        d_a = torch.where(
            a_contiguous,
            dv.gather(1, a_indices),
            torch.zeros(B, K, device=device, dtype=dv.dtype),
        ).to(dtype=torch.float32)

        # Operand B is the contiguous digit run starting after the operator.
        b_positions = b_start.unsqueeze(1) + digit_offsets
        b_in_range = b_positions < S
        b_indices = b_positions.clamp(0, S - 1)
        b_is_digit = digit_mask.gather(1, b_indices) & b_in_range
        b_contiguous = b_is_digit.long().cumprod(dim=1).bool()
        b_count = b_contiguous.long().sum(dim=1, keepdim=True)
        b_lsb_positions = b_start.unsqueeze(1) + b_count - 1 - digit_offsets
        b_lsb_in_range = b_lsb_positions >= b_start.unsqueeze(1)
        b_lsb_indices = b_lsb_positions.clamp(0, S - 1)
        d_b = torch.where(
            b_lsb_in_range,
            dv.gather(1, b_lsb_indices),
            torch.zeros(B, K, device=device, dtype=dv.dtype),
        ).to(dtype=torch.float32)

        return d_a, d_b

    def forward(
        self,
        h: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        op_pos: Tensor | None = None,
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

        if op_pos is None:
            op_pos = self._find_operator_positions(input_ids)

        if self._per_digit:
            d_a_flat, d_b_flat = self._extract_per_digit(input_ids, op_pos)
        else:
            d_a_flat, d_b_flat = self._extract_single_token(input_ids, op_pos)

        # Broadcast to all sequence positions: [B, K] -> [B, S, K]
        d_a = d_a_flat.unsqueeze(1).expand(B, S, K)
        d_b = d_b_flat.unsqueeze(1).expand(B, S, K)

        return d_a, d_b, d_a, d_b
