"""Stage 1: Deterministic Operand Extraction via Token Lookup.

Extraction is fully deterministic — no learned parameters:

1. A frozen lookup table maps every token ID to its digit vector (e.g., token
   "300" → [0, 0, 3, 0, 0, 0]). Non-number tokens map to zeros.
2. Operator tokens (+, -, *, /, ^) are identified by their token IDs.
3. Operand A is the token immediately before the operator.
   Operand B is the token immediately after the operator.
4. Digit vectors are looked up directly — no attention, no MLP.

This removes all learning from Stage 1, making extraction exact by
construction. Only Stage 4 (injection) needs to be trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# GPT-2 token IDs for arithmetic operators
_OPERATOR_TOKEN_IDS = {
    10,   # +
    12,   # -
    9,    # *
    14,   # /
    61,   # ^
}


class OperandExtractor(nn.Module):
    """Deterministic operand extraction via token ID lookup.

    No learned parameters. Scans input_ids for operator tokens,
    then looks up digit vectors for the adjacent number tokens.
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

        # Token lookup table — built by build_token_digits_table()
        self.register_buffer(
            "token_digits", torch.zeros(1, num_digits), persistent=True
        )

        # Operator token IDs as a buffer for efficient lookup
        op_ids = torch.zeros(50257, dtype=torch.bool)  # GPT-2 vocab size
        for tid in _OPERATOR_TOKEN_IDS:
            op_ids[tid] = True
        self.register_buffer("is_operator", op_ids, persistent=True)

    def build_token_digits_table(self, tokenizer) -> None:
        """Build frozen lookup: token_id -> digit vector [V, K].

        Must be called once after construction with the model's tokenizer.
        Non-number tokens get all-zero digit vectors.
        """
        vocab_size = tokenizer.vocab_size
        table = torch.zeros(vocab_size, self.num_digits)

        for token_id in range(vocab_size):
            text = tokenizer.decode([token_id]).strip()
            if text.isascii() and text.isdigit():
                value = int(text)
                for d in range(self.num_digits):
                    table[token_id, d] = value % 10
                    value //= 10

        self.token_digits = table.to(self.token_digits.device)

    def _find_operator_positions(self, input_ids: Tensor) -> Tensor:
        """Find the position of the first operator token in each sequence.

        Args:
            input_ids: [B, S]

        Returns:
            op_pos: [B] index of the first operator token (0 if none found)
        """
        B, S = input_ids.shape
        # Look up which tokens are operators: [B, S]
        ids_clamped = input_ids.clamp(0, self.is_operator.size(0) - 1)
        is_op = self.is_operator[ids_clamped]  # [B, S] bool

        # Find first operator position per sequence
        # Use argmax on the bool tensor — returns first True index, or 0 if none
        op_pos = is_op.long().argmax(dim=1)  # [B]
        return op_pos

    def forward(
        self,
        h: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Deterministic operand extraction.

        Finds the operator token, looks up digit vectors for adjacent tokens.
        Returns the same digit vector at every sequence position (broadcast)
        so downstream stages can use any position.

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

        # Find operator positions
        op_pos = self._find_operator_positions(input_ids)  # [B]

        # Operand A is at op_pos - 1, operand B is at op_pos + 1
        a_pos = (op_pos - 1).clamp(min=0)  # [B]
        b_pos = (op_pos + 1).clamp(max=S - 1)  # [B]

        # Gather the token IDs at operand positions
        a_token_ids = input_ids.gather(1, a_pos.unsqueeze(1)).squeeze(1)  # [B]
        b_token_ids = input_ids.gather(1, b_pos.unsqueeze(1)).squeeze(1)  # [B]

        # Look up digit vectors: [B, K]
        a_ids_clamped = a_token_ids.clamp(0, self.token_digits.size(0) - 1)
        b_ids_clamped = b_token_ids.clamp(0, self.token_digits.size(0) - 1)
        d_a_flat = self.token_digits[a_ids_clamped]  # [B, K]
        d_b_flat = self.token_digits[b_ids_clamped]  # [B, K]

        # Broadcast to all sequence positions: [B, K] -> [B, S, K]
        d_a = d_a_flat.unsqueeze(1).expand(B, S, K)
        d_b = d_b_flat.unsqueeze(1).expand(B, S, K)

        return d_a, d_b, d_a, d_b
