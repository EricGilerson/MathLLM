"""GPT-2 with Arithmetic Residual Blocks surgically inserted.

The base GPT-2 weights are fully frozen. ARBs are inserted after specified
transformer layers. The forward pass is manually unrolled to give precise
control over where ARBs execute.

At initialization (with zero W_proj), this model produces identical outputs
to the unmodified GPT-2.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from mathllm.arb.arb_module import ArithmeticResidualBlock
from mathllm.config import Config, load_config, save_config
from mathllm.model.utils import freeze_parameters


EXPORT_STATE_FILENAME = "model_state.pt"
EXPORT_CONFIG_FILENAME = "config.yaml"
EXPORT_BASE_MODEL_CONFIG_DIRNAME = "base_model_config"


class GPT2WithARB(nn.Module):
    """GPT-2 augmented with Arithmetic Residual Blocks.

    The base model is fully frozen. Only the ARB learned parameters
    (extraction and injection projections) are trainable.
    """

    def __init__(self, config: Config, base_model: GPT2LMHeadModel | None = None):
        super().__init__()
        self.config = config

        # Load pretrained GPT-2 with eager attention so our manually-constructed
        # attention mask works consistently across CPU, MPS, and CUDA.
        if base_model is None:
            base_model = GPT2LMHeadModel.from_pretrained(
                config.training.base_model, attn_implementation="eager"
            )
        if hasattr(base_model.config, "_attn_implementation"):
            base_model.config._attn_implementation = "eager"
        if hasattr(base_model.config, "attn_implementation"):
            base_model.config.attn_implementation = "eager"
        self.base_model = base_model
        self.hidden_dim = self.base_model.config.n_embd

        # Freeze ALL base model parameters
        freeze_parameters(self.base_model)

        # Detect GPT2Block KV-cache parameter name (renamed in transformers ~4.45)
        block_params = inspect.signature(GPT2Block.forward).parameters
        self._cache_kwarg = "past_key_values" if "past_key_values" in block_params else "layer_past"
        self._has_cache_position = "cache_position" in block_params

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
                dropout=config.arb.dropout,
                injector_init_std=config.arb.injector_init_std,
                gate_init_logit=config.arb.gate_init_logit,
            )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
    ) -> dict[str, Tensor | None]:
        """Forward pass with ARBs inserted after designated transformer layers.

        Manually unrolls the GPT-2 forward pass for precise ARB placement.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask (1 = attend, 0 = ignore)
            labels: [batch, seq_len] target token IDs for loss computation
            past_key_values: DynamicCache from prior generation steps
            use_cache: whether to return updated KV cache

        Returns:
            dict with 'loss' (if labels provided), 'logits', and optionally
            'past_key_values'
        """
        transformer = self.base_model.transformer
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Normalise cache: DynamicCache (new API) or list-of-tuples (old API)
        use_dynamic_cache = self._cache_kwarg == "past_key_values"
        if past_key_values is None:
            past_len = 0
        elif use_dynamic_cache:
            past_len = past_key_values.get_seq_length()
        else:
            # Old API: list of (key, value) tuples, one per layer
            past_len = past_key_values[0][0].size(-2) if past_key_values else 0
        cache_position = torch.arange(
            past_len, past_len + seq_len, dtype=torch.long, device=device
        )
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        inputs_embeds = transformer.wte(input_ids)
        position_embeds = transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = transformer.drop(hidden_states)

        # Build causal attention mask in GPT-2 format
        # GPT-2 expects [batch, 1, 1, total_len] with 0.0 for attend, -10000.0 for mask
        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :].to(
                dtype=hidden_states.dtype
            )
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None

        # Iterate through transformer blocks, inserting ARBs
        presents = []
        arb_extractions: dict[int, tuple[Tensor, Tensor]] = {}
        for i, block in enumerate(transformer.h):
            block_kwargs = {
                "attention_mask": extended_mask,
            }
            if use_cache:
                if use_dynamic_cache:
                    block_kwargs[self._cache_kwarg] = past_key_values
                    if self._has_cache_position:
                        block_kwargs["cache_position"] = cache_position
                else:
                    # Old API: pass this layer's (key, value) tuple or None
                    layer_cache = past_key_values[i] if past_key_values else None
                    block_kwargs[self._cache_kwarg] = layer_cache
                block_kwargs["use_cache"] = True

            block_output = block(hidden_states, **block_kwargs)
            hidden_states = block_output[0]
            if use_cache and not use_dynamic_cache:
                presents.append(block_output[1])

            # Insert ARB after this layer if configured
            if i in self.arb_positions:
                hidden_states, d_a, d_b = self.arbs[str(i)](hidden_states)
                arb_extractions[i] = (d_a, d_b)

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
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            valid_count = flat_labels.ne(-100).sum()
            if valid_count.item() == 0:
                loss = flat_logits.new_zeros(())
            else:
                loss = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                    reduction="sum",
                ) / valid_count

        result: dict[str, object] = {
            "loss": loss,
            "logits": logits,
            "arb_extractions": arb_extractions,
        }
        if use_cache:
            result["past_key_values"] = past_key_values if use_dynamic_cache else presents
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = True,
    ) -> Tensor:
        """Autoregressive text generation with KV caching.

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
        use_dynamic_cache = self._cache_kwarg == "past_key_values"
        cache = DynamicCache() if use_dynamic_cache else None

        for _ in range(max_new_tokens):
            # On first step feed full prompt; after that only the new token
            cache_len = (
                cache.get_seq_length() if use_dynamic_cache and cache
                else (cache[0][0].size(-2) if cache else 0)
            )
            if cache_len == 0:
                cur_input = generated
            else:
                cur_input = generated[:, -1:]

            outputs = self.forward(cur_input, past_key_values=cache, use_cache=True)
            cache = outputs["past_key_values"]
            next_token_logits = outputs["logits"][:, -1, :]  # [batch, vocab]

            if greedy:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                logits_scaled = next_token_logits / temperature
                if top_k is not None:
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

    def save_exported_model(self, output_dir: str | Path, tokenizer: GPT2Tokenizer) -> Path:
        """Save a self-contained, inference-ready model bundle."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.base_model.config.save_pretrained(output_dir / EXPORT_BASE_MODEL_CONFIG_DIRNAME)
        tokenizer.save_pretrained(output_dir)
        save_config(self.config, output_dir / EXPORT_CONFIG_FILENAME)
        torch.save(self.state_dict(), output_dir / EXPORT_STATE_FILENAME)
        return output_dir

    @classmethod
    def from_exported_model(
        cls,
        output_dir: str | Path,
        device: torch.device | str | None = None,
    ) -> tuple[GPT2WithARB, GPT2Tokenizer, Config]:
        """Load a model bundle produced by save_exported_model."""
        output_dir = Path(output_dir)
        config = load_config(output_dir / EXPORT_CONFIG_FILENAME)
        tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model_config = GPT2Config.from_pretrained(
            output_dir / EXPORT_BASE_MODEL_CONFIG_DIRNAME
        )
        if hasattr(base_model_config, "_attn_implementation"):
            base_model_config._attn_implementation = "eager"
        if hasattr(base_model_config, "attn_implementation"):
            base_model_config.attn_implementation = "eager"

        model = cls(config, base_model=GPT2LMHeadModel(base_model_config))
        state_dict = torch.load(
            output_dir / EXPORT_STATE_FILENAME,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state_dict)
        if device is not None:
            model.to(device)
        return model, tokenizer, config
