"""Train and evaluate fully-trainable baseline and ARB toy decoders."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from mathllm.config import ARBConfig, Config, RNSConfig, TrainingConfig
from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.pretraining.char_tokenizer import CharTokenizer
from mathllm.pretraining.data import MixtureSpec, build_mixture, load_prose_documents, save_mixture


@dataclass
class ToyModelConfig:
    n_layer: int = 4
    n_embd: int = 144
    n_head: int = 4
    n_inner: int = 576


@dataclass
class ToyDataConfig:
    mixture_file: str = "pretraining_data/toy_smoke.pt"
    prose_documents: int = 256
    train_blocks: int = 256
    eval_blocks: int = 64
    arithmetic_token_fraction: float = 0.25
    max_digits: int = 2
    invocation_fraction: float = 0.25


@dataclass
class ToyTrainingConfig:
    output_dir: str = "pretraining_runs/toy_smoke"
    device: str = "cpu"
    seed: int = 20260723
    context_length: int = 64
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 3
    eval_batches: int = 4
    eval_cases: int = 8


@dataclass
class ToyExperimentConfig:
    model: ToyModelConfig = field(default_factory=ToyModelConfig)
    data: ToyDataConfig = field(default_factory=ToyDataConfig)
    training: ToyTrainingConfig = field(default_factory=ToyTrainingConfig)


def _merge(dc, values: dict):
    for key, value in values.items():
        current = getattr(dc, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _merge(current, value)
        else:
            setattr(dc, key, value)


def load_toy_config(path: str | Path) -> ToyExperimentConfig:
    config = ToyExperimentConfig()
    with open(path) as handle:
        _merge(config, yaml.safe_load(handle) or {})
    return config


def prepare_data(config: ToyExperimentConfig) -> dict[str, object]:
    tokenizer = CharTokenizer()
    prose = load_prose_documents(config.data.prose_documents)
    spec = MixtureSpec(
        context_length=config.training.context_length,
        train_blocks=config.data.train_blocks,
        eval_blocks=config.data.eval_blocks,
        arithmetic_token_fraction=config.data.arithmetic_token_fraction,
        max_digits=config.data.max_digits,
        invocation_fraction=config.data.invocation_fraction,
        seed=config.training.seed,
    )
    mixture = build_mixture(spec, prose, tokenizer)
    save_mixture(config.data.mixture_file, mixture)
    return mixture


def _gpt2_config(tokenizer: CharTokenizer, model: ToyModelConfig, context_length: int) -> GPT2Config:
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=context_length + 1,
        n_ctx=context_length + 1,
        n_embd=model.n_embd,
        n_layer=model.n_layer,
        n_head=model.n_head,
        n_inner=model.n_inner,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config.loss_type = "ForCausalLMLoss"
    return config


def build_model(config: ToyExperimentConfig, variant: str, tokenizer: CharTokenizer) -> nn.Module:
    torch.manual_seed(config.training.seed)
    base = GPT2LMHeadModel(_gpt2_config(tokenizer, config.model, config.training.context_length))
    if variant == "baseline":
        return base
    if variant != "arb":
        raise ValueError("variant must be 'baseline' or 'arb'")
    arb_config = Config(
        rns=RNSConfig(primes=(7, 11, 13, 17, 19, 23), num_digit_slots=max(4, config.data.max_digits * 2)),
        arb=ARBConfig(
            layer_positions=(max(0, config.model.n_layer // 2 - 1), config.model.n_layer - 1),
            dropout=0.0,
            injector_init_std=0.02,
            gate_init_logit=0.0,
            extraction_mlp_hidden=64,
            injection_pos_dim=8,
            injection_mlp_hidden=64,
            injection_attn_dim=4,
            injection_hard_select=True,
            lora_rank=0,
        ),
        training=TrainingConfig(answer_only_loss=False),
    )
    wrapped = GPT2WithARB(arb_config, base_model=base, freeze_base=False)
    wrapped.build_token_digit_tables(tokenizer)
    return wrapped


def _loss(outputs) -> torch.Tensor:
    return outputs["loss"] if isinstance(outputs, dict) else outputs.loss


def _forward(model: nn.Module, sequence: torch.Tensor):
    mask = torch.ones_like(sequence)
    return model(input_ids=sequence, attention_mask=mask, labels=sequence)


def _evaluate_loss(model: nn.Module, sequences: torch.Tensor, sources: torch.Tensor, source: int, device: torch.device, batches: int, batch_size: int) -> float:
    matching = torch.where(sources == source)[0]
    if not len(matching):
        return float("nan")
    losses = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, min(len(matching), batches * batch_size), batch_size):
            batch = sequences[matching[start:start + batch_size]].to(device)
            losses.append(float(_loss(_forward(model, batch)).item()))
    return sum(losses) / max(len(losses), 1)


def _generate(model: nn.Module, tokenizer: CharTokenizer, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        if isinstance(model, GPT2WithARB):
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, greedy=True)
        else:
            output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def _arithmetic_metrics(model: nn.Module, tokenizer: CharTokenizer, config: ToyExperimentConfig, device: torch.device) -> dict[str, float]:
    from mathllm.pretraining.data import _sample_expression

    rng = random.Random(config.training.seed + 99)
    direct_correct = 0
    invocation_correct = 0
    for _ in range(config.training.eval_cases):
        a, op, b, result = _sample_expression(rng, config.data.max_digits)
        expected = str(result)
        direct = _generate(model, tokenizer, f"{a}{op}{b}=", len(expected) + 2, device)
        direct_correct += int(direct[len(f"{a}{op}{b}="):].startswith(expected))
        words = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}
        prompt = f"Compute {a} {words[op]} {b}. Equation: "
        expected_equation = f"{a}{op}{b}={result}"
        invoked = _generate(model, tokenizer, prompt, len(expected_equation) + 2, device)
        invocation_correct += int(invoked[len(prompt):].startswith(expected_equation))
    n = config.training.eval_cases
    return {"direct_arithmetic_accuracy": direct_correct / n, "equation_invocation_accuracy": invocation_correct / n}


def run_training(config: ToyExperimentConfig, variant: str, prepare: bool = False) -> dict[str, object]:
    torch.manual_seed(config.training.seed)
    random.seed(config.training.seed)
    mixture_path = Path(config.data.mixture_file)
    mixture = prepare_data(config) if prepare or not mixture_path.exists() else torch.load(mixture_path, map_location="cpu", weights_only=False)
    tokenizer = CharTokenizer()
    device = torch.device(config.training.device)
    model = build_model(config, variant, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    sequences = mixture["train_input_ids"]
    generator = torch.Generator().manual_seed(config.training.seed)
    order = torch.randperm(len(sequences), generator=generator)
    losses = []
    model.train()
    for step in range(config.training.max_steps):
        start = (step * config.training.batch_size) % len(order)
        indices = order[start:start + config.training.batch_size]
        if len(indices) < config.training.batch_size:
            indices = torch.cat([indices, order[:config.training.batch_size - len(indices)]])
        batch = sequences[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(_forward(model, batch))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))

    eval_ids = mixture["eval_input_ids"]
    eval_sources = mixture["eval_sources"]
    prose_nll = _evaluate_loss(model, eval_ids, eval_sources, 0, device, config.training.eval_batches, config.training.batch_size)
    arithmetic_nll = _evaluate_loss(model, eval_ids, eval_sources, 1, device, config.training.eval_batches, config.training.batch_size)
    metrics = {
        "train_loss": losses,
        "heldout_prose_nll": prose_nll,
        "heldout_prose_ppl": math.exp(prose_nll),
        "heldout_arithmetic_nll": arithmetic_nll,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        **_arithmetic_metrics(model, tokenizer, config, device),
    }
    output_dir = Path(config.training.output_dir) / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": asdict(config), "metrics": metrics}, output_dir / "checkpoint.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
