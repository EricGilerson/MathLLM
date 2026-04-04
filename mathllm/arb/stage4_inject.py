"""Stage 4: Learned Result Injection.

Projects concatenated result digit vectors back into the transformer hidden state
via a learned linear projection. Uses a residual connection: h' = h + W_proj(results).

The default initialization is very small but non-zero so the ARB starts close to
the frozen base model while still allowing gradients to reach the extraction
layers from the first update step.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ResultInjector(nn.Module):
    """Project arithmetic results back into the hidden state dimension.

    Concatenates the digit vectors from all operations (add, sub, mul, exp)
    and projects to hidden_dim. The residual connection is applied here.
    """

    def __init__(
        self,
        hidden_dim: int,
        result_dim: int,
        dropout: float = 0.1,
        init_std: float = 1e-3,
    ):
        """
        Args:
            hidden_dim: Transformer hidden dimension (e.g. 768 for GPT-2)
            result_dim: Total dimension of concatenated result vectors
                        (e.g. 4 * (K+1) = 4 * 11 = 44 for 4 ops with sign)
            dropout: Dropout rate on the projected delta before injection.
                     Prevents the injection from always firing on every token.
            init_std: Stddev for the projection weight init. Set to 0.0 to
                      recover exact identity-at-init behavior.
        """
        super().__init__()
        self.projection = nn.Linear(result_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Small non-zero init keeps the block near-identity while allowing
        # gradients to reach earlier ARB stages immediately.
        if init_std == 0.0:
            nn.init.zeros_(self.projection.weight)
        else:
            nn.init.normal_(self.projection.weight, mean=0.0, std=init_std)
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
