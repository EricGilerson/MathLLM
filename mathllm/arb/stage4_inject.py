"""Stage 4: Learned Result Injection.

Projects concatenated result digit vectors back into the transformer hidden state
via a learned linear projection. Uses a residual connection: h' = h + W_proj(results).

Critical: W_proj is initialized to ZERO so the ARB starts as a no-op.
The model's behavior is identical to the unmodified base at initialization.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ResultInjector(nn.Module):
    """Project arithmetic results back into the hidden state dimension.

    Concatenates the digit vectors from all operations (add, sub, mul, exp)
    and projects to hidden_dim. The residual connection is applied here.
    """

    def __init__(self, hidden_dim: int, result_dim: int, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Transformer hidden dimension (e.g. 768 for GPT-2)
            result_dim: Total dimension of concatenated result vectors
                        (e.g. 4 * (K+1) = 4 * 11 = 44 for 4 ops with sign)
            dropout: Dropout rate on the projected delta before injection.
                     Prevents the injection from always firing on every token.
        """
        super().__init__()
        self.projection = nn.Linear(result_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # CRITICAL: Initialize to zero so ARB is a no-op at start.
        # The residual connection passes h through unchanged until training
        # teaches the projection when and how to inject results.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, results: Tensor, h: Tensor) -> Tensor:
        """Inject results into hidden state via residual connection.

        Args:
            results: [batch, seq, result_dim] — concatenated result digit vectors
            h: [batch, seq, hidden_dim] — original hidden state

        Returns:
            h': [batch, seq, hidden_dim] — h + delta_h
        """
        delta_h = self.projection(results)
        delta_h = self.dropout(delta_h)
        return h + delta_h
