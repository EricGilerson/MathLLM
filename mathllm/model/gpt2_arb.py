"""GPT-2 with Arithmetic Residual Blocks surgically inserted.

The base GPT-2 weights are fully frozen. ARBs are inserted after specified
transformer layers. The forward pass is manually unrolled to give precise
control over where ARBs execute.

At initialization (with zero W_proj), this model produces identical outputs
to the unmodified GPT-2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mathllm.arb.arb_module import ArithmeticResidualBlock
from mathllm.config import Config
from mathllm.model.utils import freeze_parameters


class GPT2WithARB(nn.Module):
    """GPT-2 augmented with Arithmetic Residual Blocks.

    The base model is fully frozen. Only the ARB learned parameters
    (extraction and injection projections) are trainable.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load pretrained GPT-2
        self.base_model = GPT2LMHeadModel.from_pretrained(config.training.base_model)
        self.hidden_dim = self.base_model.config.n_embd

        # Freeze ALL base model parameters
        freeze_parameters(self.base_model)

        # Create ARBs at specified layer positions
        self.arb_positions = set(config.arb.layer_positions)
        self.arbs = nn.ModuleDict()
        for pos in config.arb.layer_positions:
            self.arbs[str(pos)] = ArithmeticResidualBlock(
                hidden_dim=self.hidden_dim,
                primes=config.rns.primes,
                num_digits=config.rns.num_digit_slots,
                num_results=config.arb.num_results,
                softmax_temperature=config.arb.softmax_temperature,
            )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Forward pass with ARBs inserted after designated transformer layers.

        Manually unrolls the GPT-2 forward pass for precise ARB placement.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask (1 = attend, 0 = ignore)
            labels: [batch, seq_len] target token IDs for loss computation

        Returns:
            dict with 'loss' (if labels provided) and 'logits'
        """
        transformer = self.base_model.transformer
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Token + position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        inputs_embeds = transformer.wte(input_ids)
        position_embeds = transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = transformer.drop(hidden_states)

        # Build causal attention mask in GPT-2 format
        # GPT-2 expects [batch, 1, 1, seq_len] with 0.0 for attend, -10000.0 for mask
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] with 1=attend, 0=pad
            # Convert to [batch, 1, 1, seq_len] additive mask
            extended_mask = attention_mask[:, None, None, :].to(
                dtype=hidden_states.dtype
            )
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None

        # Iterate through transformer blocks, inserting ARBs
        for i, block in enumerate(transformer.h):
            block_output = block(
                hidden_states,
                attention_mask=extended_mask,
            )
            hidden_states = block_output[0]

            # Insert ARB after this layer if configured
            if i in self.arb_positions:
                hidden_states = self.arbs[str(i)](hidden_states)

        # Final layer norm
        hidden_states = transformer.ln_f(hidden_states)

        # LM head
        logits = self.base_model.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = True,
    ) -> Tensor:
        """Autoregressive text generation.

        Args:
            input_ids: [batch, seq_len] prompt token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (ignored if greedy=True)
            top_k: Top-k sampling (ignored if greedy=True)
            greedy: Use greedy decoding (argmax)

        Returns:
            [batch, seq_len + generated] full sequence including prompt
        """
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # Truncate to max model length if needed
            max_len = self.base_model.config.n_positions
            input_chunk = generated[:, -max_len:]

            outputs = self.forward(input_chunk)
            next_token_logits = outputs["logits"][:, -1, :]  # [batch, vocab]

            if greedy:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                logits_scaled = next_token_logits / temperature
                if top_k is not None:
                    # Zero out everything except top-k
                    top_k_vals, _ = logits_scaled.topk(top_k, dim=-1)
                    threshold = top_k_vals[:, -1:]
                    logits_scaled = logits_scaled.masked_fill(
                        logits_scaled < threshold, float("-inf")
                    )
                probs = F.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS token
            if (next_token == self.base_model.config.eos_token_id).all():
                break

        return generated

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the ARB learned parameters for optimization."""
        params = []
        for arb in self.arbs.values():
            params.extend(p for p in arb.parameters() if p.requires_grad)
        return params
