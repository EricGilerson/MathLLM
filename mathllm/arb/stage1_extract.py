"""Stage 1: Learned Operand Extraction.

Two MLP heads read digit-level features from the hidden state, outputting
per-digit classification logits. The discrete digits for downstream stages
are obtained via argmax with a straight-through estimator for gradient flow.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from mathllm.arb.ste import ste_argmax


class OperandExtractor(nn.Module):
    """Extract two operand digit vectors from the transformer hidden state.

    Each operand is represented as K digits in base 10, least-significant first.
    Each digit is predicted as a 10-class classification problem via a small MLP.
    Dropout on the input prevents overfitting to specific hidden-state patterns.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_digits: int = 10,
        num_classes: int = 10,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.head_a = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_digits * num_classes),
        )
        self.head_b = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_digits * num_classes),
        )

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract two operand digit vectors from hidden state.

        Args:
            h: Hidden state [batch, seq_len, hidden_dim]

        Returns:
            d_a: Digit vector for operand A [batch, seq_len, K], integers in [0, 9]
            d_b: Digit vector for operand B [batch, seq_len, K], integers in [0, 9]
            logits_a: Classification logits for operand A [batch, seq_len, K, C]
            logits_b: Classification logits for operand B [batch, seq_len, K, C]
        """
        B, S, _ = h.shape
        h_drop = self.dropout(h)

        logits_a = self.head_a(h_drop).view(B, S, self.num_digits, self.num_classes)
        logits_b = self.head_b(h_drop).view(B, S, self.num_digits, self.num_classes)

        d_a = ste_argmax(logits_a)  # [B, S, K]
        d_b = ste_argmax(logits_b)  # [B, S, K]

        return d_a, d_b, logits_a, logits_b
