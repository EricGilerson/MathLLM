"""Stage 1: Learned Operand Extraction.

Two MLP heads read digit-level features from the hidden state, outputting
per-digit classification logits. The discrete digits for downstream stages
are obtained via argmax with a straight-through estimator for gradient flow.

When attention is enabled, each operand head first applies a low-rank causal
cross-attention over the sequence. This lets the extraction attend to the
actual operand token positions rather than compressing everything into the
hidden state at one position.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mathllm.arb.ste import ste_argmax


class OperandExtractor(nn.Module):
    """Extract two operand digit vectors from the transformer hidden state.

    Each operand is represented as K digits in base 10, least-significant first.
    Each digit is predicted as a 10-class classification problem via a small MLP.
    Dropout on the input prevents overfitting to specific hidden-state patterns.

    When use_attention=True, a lightweight causal attention layer lets each
    operand head attend across the full sequence to locate operand tokens,
    rather than relying solely on the hidden state at a single position.
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
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.dropout = nn.Dropout(dropout)

        if use_attention:
            self.attn_rank = attn_rank
            self.scale = attn_rank ** -0.5
            self.q_proj_a = nn.Linear(hidden_dim, attn_rank, bias=False)
            self.k_proj_a = nn.Linear(hidden_dim, attn_rank, bias=False)
            self.q_proj_b = nn.Linear(hidden_dim, attn_rank, bias=False)
            self.k_proj_b = nn.Linear(hidden_dim, attn_rank, bias=False)

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

    def _causal_attn(
        self,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        h: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Low-rank causal attention in float32 for numerical stability."""
        orig_dtype = h.dtype
        device_type = h.device.type if h.device.type in ("cuda", "mps") else "cpu"

        # Disable autocast so all ops run in float32
        with torch.amp.autocast(device_type=device_type, enabled=False):
            h_f32 = h.float()
            B, S, _ = h_f32.shape

            Q = F.linear(h_f32, q_proj.weight.float())  # [B, S, rank]
            K = F.linear(h_f32, k_proj.weight.float())  # [B, S, rank]
            attn = (Q @ K.transpose(-1, -2)) * self.scale  # [B, S, S]

            # Causal mask: prevent attending to future positions
            causal = torch.triu(
                torch.ones(S, S, device=h.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal, -1e9)

            # Padding mask: prevent attending to padding tokens
            if attention_mask is not None:
                pad_mask = (attention_mask == 0).unsqueeze(1)  # [B, 1, S]
                attn = attn.masked_fill(pad_mask, -1e9)

            attn = F.softmax(attn, dim=-1)
            out = attn @ h_f32  # [B, S, hidden_dim]

        return out.to(orig_dtype)

    def forward(
        self, h: Tensor, attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract two operand digit vectors from hidden state.

        Args:
            h: Hidden state [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] (1 = real token, 0 = padding).
                Only used when use_attention=True.

        Returns:
            d_a: Digit vector for operand A [batch, seq_len, K], integers in [0, 9]
            d_b: Digit vector for operand B [batch, seq_len, K], integers in [0, 9]
            logits_a: Classification logits for operand A [batch, seq_len, K, C]
            logits_b: Classification logits for operand B [batch, seq_len, K, C]
        """
        B, S, _ = h.shape

        if self.use_attention:
            h_a = self._causal_attn(self.q_proj_a, self.k_proj_a, h, attention_mask)
            h_b = self._causal_attn(self.q_proj_b, self.k_proj_b, h, attention_mask)
        else:
            h_a = h
            h_b = h

        h_a = self.dropout(h_a)
        h_b = self.dropout(h_b)

        logits_a = self.head_a(h_a).view(B, S, self.num_digits, self.num_classes)
        logits_b = self.head_b(h_b).view(B, S, self.num_digits, self.num_classes)

        d_a = ste_argmax(logits_a)  # [B, S, K]
        d_b = ste_argmax(logits_b)  # [B, S, K]

        return d_a, d_b, logits_a, logits_b
