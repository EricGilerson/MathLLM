"""Stage 1: Learned Operand Extraction via Token Lookup + Attention.

Instead of decoding numbers from hidden states via MLP classification, this
module uses a deterministic approach:

1. A frozen lookup table maps every token ID to its digit vector (e.g., token
   "300" → [0, 0, 3, 0, 0, 0]). Non-number tokens map to zeros.
2. Learned causal attention (separate Q/K per operand) selects which token
   positions contain operand A vs operand B.
3. The extraction output is: attention_weights @ digit_lookup[input_ids].

This reduces the learning problem from "decode 768-dim hidden states into
digit classifications" to "put attention weight on the correct token."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mathllm.arb.ste import ste_round_clamp


class OperandExtractor(nn.Module):
    """Extract two operand digit vectors using token lookup + learned attention.

    Each operand is represented as K digits in base 10, least-significant first.
    A frozen lookup table provides exact digit vectors for number tokens.
    Learned attention selects which positions to read from.
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

        self.attn_rank = attn_rank
        self.scale = attn_rank ** -0.5
        self.q_proj_a = nn.Linear(hidden_dim, attn_rank, bias=False)
        self.k_proj_a = nn.Linear(hidden_dim, attn_rank, bias=False)
        self.q_proj_b = nn.Linear(hidden_dim, attn_rank, bias=False)
        self.k_proj_b = nn.Linear(hidden_dim, attn_rank, bias=False)

        # Token lookup table will be registered by build_token_digits_table()
        # Initialized as empty — must call build_token_digits_table before use
        self.register_buffer(
            "token_digits", torch.zeros(1, num_digits), persistent=True
        )
        self._table_built = False

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
        self._table_built = True

    def _compute_attn_weights(
        self,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        h: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute causal attention weights in float32 for stability.

        Returns:
            attn_weights: [B, S, S] attention probabilities
        """
        device_type = h.device.type if h.device.type in ("cuda", "mps") else "cpu"

        with torch.amp.autocast(device_type=device_type, enabled=False):
            h_f32 = h.float()
            B, S, _ = h_f32.shape

            Q = F.linear(h_f32, q_proj.weight.float())  # [B, S, rank]
            K = F.linear(h_f32, k_proj.weight.float())  # [B, S, rank]
            attn = (Q @ K.transpose(-1, -2)) * self.scale  # [B, S, S]

            # Causal mask
            causal = torch.triu(
                torch.ones(S, S, device=h.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal, -1e9)

            # Padding mask
            if attention_mask is not None:
                pad_mask = (attention_mask == 0).unsqueeze(1)  # [B, 1, S]
                attn = attn.masked_fill(pad_mask, -1e9)

            attn = F.softmax(attn, dim=-1)  # [B, S, S]

        return attn

    def forward(
        self,
        h: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract two operand digit vectors via attention over token digits.

        Args:
            h: Hidden state [batch, seq_len, hidden_dim]
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] (1 = real token, 0 = padding)

        Returns:
            d_a: Rounded digit vector for operand A [batch, seq_len, K]
            d_b: Rounded digit vector for operand B [batch, seq_len, K]
            soft_a: Continuous (pre-round) digits for A [batch, seq_len, K]
            soft_b: Continuous (pre-round) digits for B [batch, seq_len, K]
        """
        # Look up digit vectors for each token: [B, S, K]
        # Clamp to valid range in case of out-of-vocab tokens
        ids_clamped = input_ids.clamp(0, self.token_digits.size(0) - 1)
        digit_vectors = self.token_digits[ids_clamped]  # [B, S, K]

        # Compute attention weights for each operand head
        attn_a = self._compute_attn_weights(
            self.q_proj_a, self.k_proj_a, h, attention_mask
        )  # [B, S, S]
        attn_b = self._compute_attn_weights(
            self.q_proj_b, self.k_proj_b, h, attention_mask
        )  # [B, S, S]

        # Attend over digit vectors: [B, S, S] @ [B, S, K] -> [B, S, K]
        soft_a = attn_a @ digit_vectors.float()
        soft_b = attn_b @ digit_vectors.float()

        # Round to integer digits with STE for gradient flow
        d_a = ste_round_clamp(soft_a, 0, 9)
        d_b = ste_round_clamp(soft_b, 0, 9)

        return d_a, d_b, soft_a, soft_b
