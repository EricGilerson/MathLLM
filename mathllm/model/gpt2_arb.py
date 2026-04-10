"""Transformer with Arithmetic Residual Blocks surgically inserted.

Supports both GPT-2 and LLaMA-style (e.g. SmolLM2-135M) base models.
The base model weights are fully frozen. ARBs are inserted after specified
transformer layers. The forward pass is manually unrolled to give precise
control over where ARBs execute.

At initialization (with zero W_proj), this model produces identical outputs
to the unmodified base model.
"""

from __future__ import annotations

import inspect
from enum import Enum, auto
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.cache_utils import DynamicCache

import logging

logger = logging.getLogger(__name__)

from mathllm.arb.arb_module import ArithmeticResidualBlock
from mathllm.config import Config, load_config, save_config
from mathllm.model.lora import LoRALinear
from mathllm.model.utils import freeze_parameters


EXPORT_STATE_FILENAME = "model_state.pt"
EXPORT_CONFIG_FILENAME = "config.yaml"
EXPORT_BASE_MODEL_CONFIG_DIRNAME = "base_model_config"


class ModelArch(Enum):
    """Supported base model architectures."""
    GPT2 = auto()
    LLAMA = auto()


def _detect_arch(model: PreTrainedModel) -> ModelArch:
    """Auto-detect the architecture family from a HuggingFace model."""
    model_type = getattr(model.config, "model_type", "").lower()
    class_name = type(model).__name__.lower()

    if model_type == "gpt2" or "gpt2" in class_name:
        return ModelArch.GPT2
    if model_type in ("llama", "mistral", "gemma", "qwen2") or "llama" in class_name:
        return ModelArch.LLAMA

    raise ValueError(
        f"Unsupported model architecture: model_type={model_type!r}, "
        f"class={type(model).__name__}. Supported: GPT-2, LLaMA-style."
    )


def _get_hidden_dim(model: PreTrainedModel, arch: ModelArch) -> int:
    """Extract hidden dimension from model config."""
    cfg = model.config
    if arch == ModelArch.GPT2:
        return cfg.n_embd
    # LLaMA-style
    return cfg.hidden_size


class TransformerWithARB(nn.Module):
    """Transformer augmented with Arithmetic Residual Blocks.

    The base model is fully frozen. Only the ARB learned parameters
    (extraction and injection projections) are trainable.

    Supports GPT-2 and LLaMA-style (LlamaForCausalLM, MistralForCausalLM, etc.)
    base models. Architecture is auto-detected from the loaded model.
    """

    def __init__(
        self,
        config: Config,
        base_model: PreTrainedModel | None = None,
        eq_token_id: int | None = None,
    ):
        super().__init__()
        self.config = config

        # Load pretrained model if not provided
        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.training.base_model,
                attn_implementation="eager",
                torch_dtype=torch.float32,
            )

        # Force eager attention for reproducibility
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(base_model.config, attr):
                setattr(base_model.config, attr, "eager")

        self.base_model = base_model
        self.arch = _detect_arch(base_model)
        self.hidden_dim = _get_hidden_dim(base_model, self.arch)

        # Store eq_token_id; default depends on architecture
        if eq_token_id is not None:
            self._eq_token_id = eq_token_id
        elif self.arch == ModelArch.GPT2:
            self._eq_token_id = 28  # GPT-2 default for '='
        else:
            # Temporary default; overwritten by build_token_digit_tables()
            self._eq_token_id = 28

        # Freeze ALL base model parameters
        freeze_parameters(self.base_model)

        # Architecture-specific setup
        if self.arch == ModelArch.GPT2:
            self._setup_gpt2()
        else:
            self._setup_llama()

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
                num_classes=config.arb.extraction_num_classes,
                mlp_hidden=config.arb.extraction_mlp_hidden,
                use_attention=config.arb.extraction_use_attention,
                attn_rank=config.arb.extraction_attn_rank,
                eq_token_id=self._eq_token_id,
                injection_pos_dim=config.arb.injection_pos_dim,
                injection_mlp_hidden=config.arb.injection_mlp_hidden,
            )

        # LoRA adapter on LM head (optional)
        lora_rank = getattr(config.arb, "lora_rank", 0)
        if lora_rank > 0:
            lora_alpha = getattr(config.arb, "lora_alpha", 1.0)
            self.lora_head = LoRALinear(
                self.base_model.lm_head, rank=lora_rank, alpha=lora_alpha
            )
            logger.info(
                "LoRA head created: rank=%d, alpha=%.1f, params=%d",
                lora_rank, lora_alpha,
                sum(p.numel() for p in self.lora_head.parameters() if p.requires_grad),
            )
        else:
            self.lora_head = None

    def _setup_gpt2(self) -> None:
        """GPT-2-specific forward pass setup."""
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        block_params = inspect.signature(GPT2Block.forward).parameters
        self._cache_kwarg = (
            "past_key_values" if "past_key_values" in block_params else "layer_past"
        )
        self._has_cache_position = "cache_position" in block_params

    def _setup_llama(self) -> None:
        """LLaMA-style setup (also covers Mistral, Gemma, Qwen2)."""
        # Newer transformers uses past_key_values (plural)
        self._cache_kwarg = "past_key_values"
        self._has_cache_position = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_eq_token_id(self, eq_token_id: int) -> None:
        """Update the '=' token ID on all ARBs."""
        self._eq_token_id = eq_token_id
        for arb in self.arbs.values():
            arb.eq_token_id = eq_token_id

    def build_token_digit_tables(self, tokenizer) -> None:
        """Build frozen digit lookup tables in all ARB extractors.

        Also auto-detects the '=' token ID from the tokenizer.
        """
        # Auto-detect eq_token_id from tokenizer
        eq_ids = tokenizer.encode("=", add_special_tokens=False)
        if eq_ids:
            self.set_eq_token_id(eq_ids[0])
            logger.info("Auto-detected eq_token_id=%d from tokenizer", eq_ids[0])

        for key, arb in self.arbs.items():
            arb.extract.build_token_digits_table(tokenizer)
        logger.info(
            "Built token digit tables for %d ARBs (vocab_size=%d)",
            len(self.arbs),
            tokenizer.vocab_size,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
    ) -> dict[str, Tensor | None]:
        """Forward pass with ARBs inserted after designated transformer layers.

        Manually unrolls the forward pass for precise ARB placement.

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
        if self.arch == ModelArch.GPT2:
            return self._forward_gpt2(
                input_ids, attention_mask, labels, past_key_values, use_cache
            )
        else:
            return self._forward_llama(
                input_ids, attention_mask, labels, past_key_values, use_cache
            )

    def _forward_gpt2(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        labels: Tensor | None,
        past_key_values: DynamicCache | None,
        use_cache: bool,
    ) -> dict[str, Tensor | None]:
        """GPT-2 forward pass with ARBs."""
        transformer = self.base_model.transformer
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Normalise cache
        use_dynamic_cache = self._cache_kwarg == "past_key_values"
        if past_key_values is None:
            past_len = 0
        elif use_dynamic_cache:
            past_len = past_key_values.get_seq_length()
        else:
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
                    layer_cache = past_key_values[i] if past_key_values else None
                    block_kwargs[self._cache_kwarg] = layer_cache
                block_kwargs["use_cache"] = True

            block_output = block(hidden_states, **block_kwargs)
            hidden_states = block_output[0]
            if use_cache and not use_dynamic_cache:
                presents.append(block_output[1])

            # Insert ARB after this layer if configured
            if i in self.arb_positions:
                hidden_states, d_a, d_b = self.arbs[str(i)](
                    hidden_states, input_ids, attention_mask
                )
                arb_extractions[i] = (d_a, d_b)

        # Final layer norm + LM head (with optional LoRA)
        hidden_states = transformer.ln_f(hidden_states)
        if self.lora_head is not None:
            logits = self.lora_head(hidden_states)
        else:
            logits = self.base_model.lm_head(hidden_states)

        loss = self._compute_loss(logits, labels)

        result: dict[str, object] = {
            "loss": loss,
            "logits": logits,
            "arb_extractions": arb_extractions,
        }
        if use_cache:
            result["past_key_values"] = (
                past_key_values if use_dynamic_cache else presents
            )
        return result

    def _forward_llama(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        labels: Tensor | None,
        past_key_values: DynamicCache | None,
        use_cache: bool,
    ) -> dict[str, Tensor | None]:
        """LLaMA-style forward pass with ARBs.

        Works with LlamaForCausalLM, MistralForCausalLM, and similar architectures
        that use model.model.layers / model.model.norm / model.model.embed_tokens.
        """
        inner_model = self.base_model.model  # the LlamaModel inside LlamaForCausalLM
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Compute past length from cache
        if past_key_values is None:
            past_len = 0
            cache = DynamicCache() if use_cache else None
        else:
            past_len = past_key_values.get_seq_length()
            cache = past_key_values

        # Position IDs
        position_ids = torch.arange(
            past_len, past_len + seq_len, dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        hidden_states = inner_model.embed_tokens(input_ids)

        # Compute RoPE position embeddings once (shared across all layers)
        # Newer transformers versions compute these externally and pass them in
        position_embeddings = None
        if hasattr(inner_model, "rotary_emb"):
            position_embeddings = inner_model.rotary_emb(
                hidden_states, position_ids
            )

        # Prepare 4D causal mask if attention_mask is provided
        prepared_mask = None
        if attention_mask is not None:
            prepared_mask = self._prepare_llama_mask(
                attention_mask, hidden_states, past_len
            )

        # Iterate through decoder layers, inserting ARBs
        arb_extractions: dict[int, tuple[Tensor, Tensor]] = {}
        for i, layer in enumerate(inner_model.layers):
            layer_kwargs: dict[str, object] = {
                "position_ids": position_ids,
            }
            if prepared_mask is not None:
                layer_kwargs["attention_mask"] = prepared_mask
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            if use_cache:
                layer_kwargs["past_key_values"] = cache
                layer_kwargs["use_cache"] = True

            layer_output = layer(hidden_states, **layer_kwargs)
            # Newer transformers returns a plain Tensor; older returns a tuple
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

            # Insert ARB after this layer if configured
            if i in self.arb_positions:
                hidden_states, d_a, d_b = self.arbs[str(i)](
                    hidden_states, input_ids, attention_mask
                )
                arb_extractions[i] = (d_a, d_b)

        # Final RMSNorm + LM head (with optional LoRA)
        hidden_states = inner_model.norm(hidden_states)
        if self.lora_head is not None:
            logits = self.lora_head(hidden_states)
        else:
            logits = self.base_model.lm_head(hidden_states)

        loss = self._compute_loss(logits, labels)

        result: dict[str, object] = {
            "loss": loss,
            "logits": logits,
            "arb_extractions": arb_extractions,
        }
        if use_cache:
            result["past_key_values"] = cache
        return result

    def _prepare_llama_mask(
        self,
        attention_mask: Tensor,
        hidden_states: Tensor,
        past_len: int,
    ) -> Tensor:
        """Prepare a 4D causal attention mask for LLaMA-style models.

        LLaMA layers expect either a 2D [B, total_len] mask or a 4D
        [B, 1, Q, KV] mask. We build the 4D version to properly handle
        both padding and causal masking with KV cache.

        Args:
            attention_mask: [B, total_len] with 1=attend, 0=pad
            hidden_states: current hidden states (for dtype)
            past_len: length of cached KV sequence

        Returns:
            4D causal mask [B, 1, seq_len, total_len]
        """
        B, total_len = attention_mask.shape
        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype
        device = attention_mask.device

        # Start with the padding mask expanded to 4D
        # [B, 1, 1, total_len] -- broadcast over query positions
        expanded_mask = attention_mask[:, None, None, :].to(dtype=dtype)

        # Build causal mask: each query position can only attend to
        # positions <= its own absolute position
        # Query positions are [past_len, past_len + seq_len)
        # Key positions are [0, total_len)
        query_pos = torch.arange(
            past_len, past_len + seq_len, device=device
        ).unsqueeze(1)  # [seq_len, 1]
        key_pos = torch.arange(total_len, device=device).unsqueeze(0)  # [1, total_len]
        causal = (key_pos <= query_pos).to(dtype=dtype)  # [seq_len, total_len]
        causal = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, total_len]

        # Combine: attend only where both causal and padding allow
        combined = expanded_mask * causal  # [B, 1, seq_len, total_len]

        # Convert to additive mask: 0.0 for attend, large negative for mask
        mask_value = torch.finfo(dtype).min
        return combined.masked_fill(combined == 0.0, mask_value).masked_fill(
            combined != 0.0, 0.0
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_loss(logits: Tensor, labels: Tensor | None) -> Tensor | None:
        """Compute next-token prediction loss if labels provided."""
        if labels is None:
            return None
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        valid_count = flat_labels.ne(-100).sum()
        if valid_count.item() == 0:
            return flat_logits.new_zeros(())
        return F.cross_entropy(
            flat_logits.float(), flat_labels, ignore_index=-100, reduction="sum",
        ) / valid_count

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

        Uses ARB generation cache mode so that arithmetic results computed
        from the full prompt are reused on subsequent steps (fixing the bug
        where step 2+ only sees the last token and can't find the operator).

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
        cache = DynamicCache()

        # Enter generation cache mode for all ARBs
        for arb in self.arbs.values():
            arb.enter_generation_mode()

        try:
            for _ in range(max_new_tokens):
                cache_len = cache.get_seq_length() if cache else 0
                if cache_len == 0:
                    cur_input = generated
                else:
                    cur_input = generated[:, -1:]

                # Build attention mask for the full sequence so far
                # (needed for proper causal masking with KV cache)
                full_len = generated.shape[1]
                attn_mask = torch.ones(
                    generated.shape[0], full_len,
                    dtype=torch.long, device=generated.device,
                )

                outputs = self.forward(
                    cur_input,
                    attention_mask=attn_mask,
                    past_key_values=cache,
                    use_cache=True,
                )
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
        finally:
            # Always exit generation mode to clean up cached state
            for arb in self.arbs.values():
                arb.exit_generation_mode()

        return generated

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the ARB learned parameters (and LoRA if active) for optimization."""
        params = []
        for arb in self.arbs.values():
            params.extend(p for p in arb.parameters() if p.requires_grad)
        if self.lora_head is not None:
            params.extend(p for p in self.lora_head.parameters() if p.requires_grad)
        return params

    def save_exported_model(
        self, output_dir: str | Path, tokenizer: PreTrainedTokenizerBase
    ) -> Path:
        """Save a self-contained, inference-ready model bundle."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.base_model.config.save_pretrained(
            output_dir / EXPORT_BASE_MODEL_CONFIG_DIRNAME
        )
        tokenizer.save_pretrained(output_dir)
        save_config(self.config, output_dir / EXPORT_CONFIG_FILENAME)
        torch.save(self.state_dict(), output_dir / EXPORT_STATE_FILENAME)
        return output_dir

    @classmethod
    def from_exported_model(
        cls,
        output_dir: str | Path,
        device: torch.device | str | None = None,
    ) -> tuple[TransformerWithARB, PreTrainedTokenizerBase, Config]:
        """Load a model bundle produced by save_exported_model."""
        output_dir = Path(output_dir)
        config = load_config(output_dir / EXPORT_CONFIG_FILENAME)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model_config = AutoConfig.from_pretrained(
            output_dir / EXPORT_BASE_MODEL_CONFIG_DIRNAME
        )
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(base_model_config, attr):
                setattr(base_model_config, attr, "eager")

        base_model = AutoModelForCausalLM.from_config(base_model_config)

        # Detect eq_token_id from tokenizer
        eq_ids = tokenizer.encode("=", add_special_tokens=False)
        eq_token_id = eq_ids[0] if eq_ids else 28

        model = cls(config, base_model=base_model, eq_token_id=eq_token_id)
        state_dict = torch.load(
            output_dir / EXPORT_STATE_FILENAME,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state_dict)
        if device is not None:
            model.to(device)
        return model, tokenizer, config


# Backward compatibility alias
GPT2WithARB = TransformerWithARB
