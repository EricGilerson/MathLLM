"""Stage 4: Learned Result Injection.

Projects concatenated result digit vectors back into the transformer hidden state
via a learned projection. Uses a residual connection: h' = h + gate * proj(results).

Supports both linear and MLP projection modes. The MLP provides nonlinear
capacity to produce stronger, more targeted perturbations that can override
the frozen base model's existing biases.
"""

from __future__ import annotations

import torch
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
        gate_init_logit: float = -2.0,
        mlp_hidden: int = 0,
    ):
        """
        Args:
            hidden_dim: Transformer hidden dimension (e.g. 768 for GPT-2)
            result_dim: Total dimension of concatenated result vectors
            dropout: Dropout rate on the projected delta before injection.
            init_std: Stddev for the final projection weight init.
            gate_init_logit: Initial logit for the learnable injection gate.
            mlp_hidden: Hidden dim for MLP injection. 0 = linear projection.
        """
        super().__init__()
        self.gate_logit = nn.Parameter(torch.tensor(gate_init_logit))
        self.dropout = nn.Dropout(dropout)

        if mlp_hidden > 0:
            self.projection = nn.Sequential(
                nn.Linear(result_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, hidden_dim),
            )
            # Init: first layer with reasonable scale, final layer small
            nn.init.kaiming_normal_(self.projection[0].weight)
            nn.init.zeros_(self.projection[0].bias)
            if init_std == 0.0:
                nn.init.zeros_(self.projection[2].weight)
            else:
                nn.init.normal_(self.projection[2].weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.projection[2].bias)
        else:
            self.projection = nn.Linear(result_dim, hidden_dim)
            if init_std == 0.0:
                nn.init.zeros_(self.projection.weight)
            else:
                nn.init.normal_(self.projection.weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.projection.bias)

    def forward(self, results: Tensor, h: Tensor) -> Tensor:
        """Inject results into hidden state via gated residual connection.

        Args:
            results: [batch, seq, result_dim] — concatenated result digit vectors
            h: [batch, seq, hidden_dim] — original hidden state

        Returns:
            h': [batch, seq, hidden_dim] — h + gate * delta_h
        """
        # Get the weight tensor for dtype reference (works for both Linear and Sequential)
        if isinstance(self.projection, nn.Sequential):
            proj_dtype = self.projection[0].weight.dtype
        else:
            proj_dtype = self.projection.weight.dtype

        results = results.to(dtype=proj_dtype)
        gate = torch.sigmoid(self.gate_logit)
        delta_h = self.projection(results)
        delta_h = self.dropout(delta_h)
        return h + gate * delta_h.to(dtype=h.dtype)
