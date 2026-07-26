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
from tqdm.auto import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from mathllm.config import ARBConfig, Config, RNSConfig, TrainingConfig
from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.pretraining.arithmetic_bpe_tokenizer import ArithmeticBPETokenizer
from mathllm.pretraining.data import (
    MixtureSpec,
    arithmetic_texts,
    build_mixture,
    contextual_arithmetic_texts,
    load_prose_documents,
    save_mixture,
)


@dataclass
class ToyModelConfig:
    n_layer: int = 4
    n_embd: int = 144
    n_head: int = 4
    n_inner: int = 576
    # Optional MLP width for the plain-LM capacity control.  This lets the
    # baseline match ARB's interface parameters without changing its hidden
    # state width, tokenizer, data, or training path.
    baseline_n_inner: int | None = None


@dataclass
class ToyDataConfig:
    mixture_file: str = "pretraining_data/toy_smoke.pt"
    tokenizer_file: str = "pretraining_data/toy_smoke_tokenizer.json"
    tokenizer_vocab_size: int = 1024
    prose_documents: int = 256
    train_blocks: int = 256
    eval_blocks: int = 64
    arithmetic_token_fraction: float = 0.25
    max_digits: int = 2
    invocation_fraction: float = 0.25
    direct_equation_token_fraction: float | None = None
    contextual_equation_token_fraction: float | None = None
    fact_token_fraction: float = 0.0
    fact_count: int = 0
    fact_format: str = "natural"
    require_wikitext: bool = False
    require_external_prose: bool = False
    require_unique_source_blocks: bool = False
    prose_source: str = "wikitext"
    prose_source_config: str | None = None


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
    log_every: int = 25
    eval_batches: int = 4
    eval_cases: int = 8
    # Optional intermediate held-out prose curve. Zero means evaluate only at
    # the end, preserving the lightweight smoke-test behavior.
    eval_every: int = 0


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
    prose = load_prose_documents(
        config.data.prose_documents,
        require_wikitext=config.data.require_wikitext,
        require_external_prose=config.data.require_external_prose,
        source=config.data.prose_source,
        source_config=config.data.prose_source_config,
    )
    # Train BPE on the same distribution family, while the mixture builder
    # itself keeps train/eval source texts disjoint.
    tokenizer_text_count = max(4096, config.data.prose_documents)
    tokenizer_training_texts = prose + arithmetic_texts(
        tokenizer_text_count,
        config.training.seed,
        config.data.max_digits,
        0.0,
    ) + contextual_arithmetic_texts(
        tokenizer_text_count,
        config.training.seed + 1,
        config.data.max_digits,
    )
    atomic_tokens: list[str] = []
    if config.data.fact_token_fraction:
        if config.data.fact_format == "atomic":
            from mathllm.pretraining.fact_benchmark import (
                atomic_fact_special_tokens,
                atomic_fact_training_texts,
                make_atomic_facts,
            )
            facts = make_atomic_facts(config.data.fact_count, config.training.seed + 5)
            atomic_tokens = atomic_fact_special_tokens(facts)
            tokenizer_training_texts += atomic_fact_training_texts(
                tokenizer_text_count, config.training.seed + 6, facts,
            )
        else:
            from mathllm.pretraining.fact_benchmark import fact_training_texts, make_facts
            tokenizer_training_texts += fact_training_texts(
                tokenizer_text_count, config.training.seed + 6,
                make_facts(config.data.fact_count, config.training.seed + 5),
            )
    tokenizer = ArithmeticBPETokenizer.train(
        tokenizer_training_texts,
        config.data.tokenizer_vocab_size,
        atomic_tokens=atomic_tokens,
    )
    tokenizer.save(config.data.tokenizer_file)
    spec = MixtureSpec(
        context_length=config.training.context_length,
        train_blocks=config.data.train_blocks,
        eval_blocks=config.data.eval_blocks,
        arithmetic_token_fraction=config.data.arithmetic_token_fraction,
        max_digits=config.data.max_digits,
        invocation_fraction=config.data.invocation_fraction,
        seed=config.training.seed,
        direct_equation_token_fraction=config.data.direct_equation_token_fraction,
        contextual_equation_token_fraction=config.data.contextual_equation_token_fraction,
        fact_token_fraction=config.data.fact_token_fraction,
        fact_count=config.data.fact_count,
        fact_format=config.data.fact_format,
        prose_source=config.data.prose_source,
        prose_source_config=config.data.prose_source_config,
        require_wikitext=config.data.require_wikitext,
        require_external_prose=config.data.require_external_prose,
        require_unique_source_blocks=config.data.require_unique_source_blocks,
    )
    mixture = build_mixture(spec, prose, tokenizer)
    save_mixture(config.data.mixture_file, mixture)
    return mixture


def _gpt2_config(
    tokenizer: ArithmeticBPETokenizer,
    model: ToyModelConfig,
    context_length: int,
    *,
    n_inner: int | None = None,
) -> GPT2Config:
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=context_length + 1,
        n_ctx=context_length + 1,
        n_embd=model.n_embd,
        n_layer=model.n_layer,
        n_head=model.n_head,
        n_inner=model.n_inner if n_inner is None else n_inner,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config.loss_type = "ForCausalLM"
    return config


def build_model(config: ToyExperimentConfig, variant: str, tokenizer: ArithmeticBPETokenizer) -> nn.Module:
    torch.manual_seed(config.training.seed)
    baseline_n_inner = config.model.baseline_n_inner or config.model.n_inner
    base = GPT2LMHeadModel(
        _gpt2_config(
            tokenizer,
            config.model,
            config.training.context_length,
            n_inner=baseline_n_inner if variant == "baseline" else config.model.n_inner,
        )
    )
    # Transformers reads this from the model instance rather than config on
    # recent versions; set both to retain the explicit causal-LM objective.
    base.loss_type = "ForCausalLM"
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


def _evaluate_loss(model: nn.Module, sequences: torch.Tensor, sources: torch.Tensor, source: int | tuple[int, ...], device: torch.device, batches: int, batch_size: int) -> float:
    if isinstance(source, tuple):
        matching = torch.where(torch.isin(sources, torch.tensor(source, device=sources.device)))[0]
    else:
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


def _generate(model: nn.Module, tokenizer: ArithmeticBPETokenizer, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        if isinstance(model, GPT2WithARB):
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, greedy=True)
        else:
            output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def _arithmetic_metrics(model: nn.Module, tokenizer: ArithmeticBPETokenizer, config: ToyExperimentConfig, device: torch.device) -> dict[str, float]:
    from mathllm.pretraining.data import (
        CONTEXTUAL_HELDOUT_TEMPLATES,
        CONTEXTUAL_TRAINING_TEMPLATES,
        _sample_expression,
    )

    rng = random.Random(config.training.seed + 99)
    direct_correct = 0
    contextual_seen_correct = 0
    contextual_unseen_correct = 0
    for _ in range(config.training.eval_cases):
        a, op, b, result = _sample_expression(rng, config.data.max_digits)
        expected = str(result)
        direct = _generate(model, tokenizer, f"{a}{op}{b}=", len(expected) + 2, device)
        direct_correct += int(direct[len(f"{a}{op}{b}="):].startswith(expected))
        # Sampling from each template family keeps evaluation practical while
        # making both metrics reflect several wordings.
        contextual_prompts = (
            (rng.choice(CONTEXTUAL_TRAINING_TEMPLATES), True),
            (rng.choice(CONTEXTUAL_HELDOUT_TEMPLATES), False),
        )
        for prefix, seen_template in contextual_prompts:
            prompt = f"{prefix}{a}{op}{b}="
            contextual = _generate(model, tokenizer, prompt, len(expected) + 2, device)
            correct = int(contextual[len(prompt):].startswith(expected))
            if seen_template:
                contextual_seen_correct += correct
            else:
                contextual_unseen_correct += correct
    n = config.training.eval_cases
    return {
        "direct_arithmetic_accuracy": direct_correct / n,
        "contextual_seen_arithmetic_accuracy": contextual_seen_correct / n,
        "contextual_unseen_arithmetic_accuracy": contextual_unseen_correct / n,
    }


def _fact_metrics(model, tokenizer, cases, device, *, atomic: bool = False) -> dict[str, float]:
    if not cases:
        return {"heldout_fact_accuracy": float("nan"), "fact_eval_cases": 0}
    if atomic:
        # Every query ends immediately before one opaque value token.  Thus
        # teacher-forced top-1 and one-token greedy decoding are the same
        # decision, without free-form length or whitespace confounds.
        prompts, expected = zip(*cases)
        encoded = [tokenizer.encode(prompt) for prompt in prompts]
        if len({len(ids) for ids in encoded}) != 1:
            raise ValueError("Atomic fact prompts must have a fixed token length")
        target_ids = torch.tensor([tokenizer.encode(value) for value in expected], device=device)
        if target_ids.ndim != 2 or target_ids.shape[1] != 1:
            raise ValueError("Atomic fact values must each encode to exactly one token")
        inputs = torch.tensor(encoded, dtype=torch.long, device=device)
        model.eval()
        with torch.inference_mode():
            outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            answer_logits = logits[:, -1, :]
            answer_nll = F.cross_entropy(answer_logits, target_ids[:, 0])
            prediction = answer_logits.argmax(dim=-1)
        correct = float((prediction == target_ids[:, 0]).float().mean().item())
        return {
            # Retain the existing key for paired-run summaries.  These two
            # accuracy names are deliberately explicit about the protocol.
            "heldout_fact_accuracy": correct,
            "fact_teacher_forced_top1_accuracy": correct,
            "fact_greedy_one_token_accuracy": correct,
            "fact_answer_nll": float(answer_nll.item()),
            "fact_eval_cases": len(cases),
        }
    correct = 0
    for prompt, expected in cases:
        full = _generate(model, tokenizer, prompt, len(expected) + 2, device)
        correct += int(full[len(prompt):].strip().startswith(expected))
    return {"heldout_fact_accuracy": correct / len(cases), "fact_eval_cases": len(cases)}


def resolve_device(requested: str) -> torch.device:
    """Choose CUDA, then Apple MPS, then CPU for an ``auto`` request."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch build")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def run_training(config: ToyExperimentConfig, variant: str, prepare: bool = False) -> dict[str, object]:
    torch.manual_seed(config.training.seed)
    random.seed(config.training.seed)
    mixture_path = Path(config.data.mixture_file)
    mixture = prepare_data(config) if prepare or not mixture_path.exists() else torch.load(mixture_path, map_location="cpu", weights_only=False)
    tokenizer = ArithmeticBPETokenizer.from_file(config.data.tokenizer_file)
    device = resolve_device(config.training.device)
    model = build_model(config, variant, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    sequences = mixture["train_input_ids"]
    eval_ids = mixture["eval_input_ids"]
    eval_sources = mixture["eval_sources"]
    generator = torch.Generator().manual_seed(config.training.seed)
    order = torch.randperm(len(sequences), generator=generator)
    losses = []
    prose_eval_history = []
    model.train()
    progress = tqdm(
        range(config.training.max_steps),
        desc=f"{variant} training ({device.type})",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:
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
        loss_value = float(loss.item())
        losses.append(loss_value)
        if config.training.eval_every and (step + 1) % config.training.eval_every == 0:
            prose_nll = _evaluate_loss(
                model, eval_ids, eval_sources, 0, device,
                config.training.eval_batches, config.training.batch_size,
            )
            prose_eval_history.append({
                "step": step + 1,
                "heldout_prose_nll": prose_nll,
                "heldout_prose_ppl": math.exp(prose_nll),
            })
            model.train()
        if (step + 1) % config.training.log_every == 0 or step == 0 or step + 1 == config.training.max_steps:
            progress.set_postfix(loss=f"{loss_value:.4f}")

    prose_nll = _evaluate_loss(model, eval_ids, eval_sources, 0, device, config.training.eval_batches, config.training.batch_size)
    direct_arithmetic_nll = _evaluate_loss(model, eval_ids, eval_sources, 1, device, config.training.eval_batches, config.training.batch_size)
    contextual_arithmetic_nll = _evaluate_loss(model, eval_ids, eval_sources, 2, device, config.training.eval_batches, config.training.batch_size)
    fact_nll = _evaluate_loss(model, eval_ids, eval_sources, 3, device, config.training.eval_batches, config.training.batch_size)
    metrics = {
        "train_loss": losses,
        "heldout_prose_history": prose_eval_history,
        "heldout_prose_nll": prose_nll,
        "heldout_prose_ppl": math.exp(prose_nll),
        # Retain the original key for scripts that consumed the legacy
        # prose/direct-arithmetic mixture.  In a three-way run this is the
        # canonical direct-equation slice, not an aggregate over both formats.
        "heldout_arithmetic_nll": direct_arithmetic_nll,
        "heldout_direct_arithmetic_nll": direct_arithmetic_nll,
        "heldout_contextual_arithmetic_nll": contextual_arithmetic_nll,
        "heldout_fact_nll": fact_nll,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        **_arithmetic_metrics(model, tokenizer, config, device),
        **_fact_metrics(
            model,
            tokenizer,
            mixture.get("fact_eval_cases", []),
            device,
            atomic=config.data.fact_format == "atomic",
        ),
    }
    output_dir = Path(config.training.output_dir) / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": asdict(config), "metrics": metrics}, output_dir / "checkpoint.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
