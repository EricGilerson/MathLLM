"""Stage 1: Learned Operand Extraction.

Two linear projections read digit-level features from the hidden state.
The continuous outputs are rounded to integers via STE for exact digit values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mathllm.arb.ste import ste_round_clamp


class OperandExtractor(nn.Module):
    """Extract two operand digit vectors from the transformer hidden state.

    Each operand is represented as K digits in base 10, least-significant first.
    The projections are learned; rounding uses a straight-through estimator.
    Dropout on the input prevents overfitting to specific hidden-state patterns.
    """

    def __init__(self, hidden_dim: int, num_digits: int = 10, dropout: float = 0.1):
        super().__init__()
        self.num_digits = num_digits
        self.dropout = nn.Dropout(dropout)
        self.W_a = nn.Linear(hidden_dim, num_digits)
        self.W_b = nn.Linear(hidden_dim, num_digits)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract two operand digit vectors from hidden state.

        Args:
            h: Hidden state [batch, seq_len, hidden_dim]

        Returns:
            d_a: Digit vector for operand A [batch, seq_len, K], integers in [0, 9]
            d_b: Digit vector for operand B [batch, seq_len, K], integers in [0, 9]
            d_a_cont: Continuous (pre-round) predictions for operand A [batch, seq_len, K]
            d_b_cont: Continuous (pre-round) predictions for operand B [batch, seq_len, K]
        """
        h_drop = self.dropout(h)
        d_a_cont = self.W_a(h_drop)
        d_b_cont = self.W_b(h_drop)

        d_a = ste_round_clamp(d_a_cont, low=0, high=9)
        d_b = ste_round_clamp(d_b_cont, low=0, high=9)

        return d_a, d_b, d_a_cont, d_b_cont
