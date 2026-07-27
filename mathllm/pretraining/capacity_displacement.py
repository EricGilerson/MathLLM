"""Fast controlled test of arithmetic-to-knowledge capacity displacement.

Unlike the broad toy-pretraining mixture, this experiment optimizes only the
two outputs relevant to the mechanism claim: opaque fact values and arithmetic
answer digits.  It compares a fact-only ceiling, a plain multitask decoder, an
ARB multitask decoder, and an architecture-matched ARB whose computed result is
zeroed.  The latter is a causal control for the extra interface architecture.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from tqdm.auto import tqdm

from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.pretraining.arithmetic_bpe_tokenizer import ArithmeticBPETokenizer
from mathllm.pretraining.data import _sample_expression
from mathllm.pretraining.experiment import (
    ToyDataConfig,
    ToyExperimentConfig,
    ToyModelConfig,
    ToyTrainingConfig,
    build_model,
    resolve_device,
)
from mathllm.pretraining.fact_benchmark import (
    compositional_fact_eval_cases,
    compositional_fact_special_tokens,
    make_compositional_facts,
)


CONDITIONS = (
    "fact_only",
    "baseline_mixed",
    "arb_mixed",
    "zero_arb_mixed",
)


@dataclass
class DisplacementDataConfig:
    dataset_file: str = "pretraining_data/capacity_displacement_5m.pt"
    tokenizer_file: str = "pretraining_data/capacity_displacement_5m_tokenizer.json"
    tokenizer_vocab_size: int = 2048
    fact_count: int = 25_000
    fact_key_vocab_size: int = 512
    fact_value_vocab_size: int = 512
    max_digits: int = 3
    arithmetic_pool_size: int = 50_000
    arithmetic_eval_cases: int = 1_000


@dataclass
class DisplacementTrainingConfig:
    output_dir: str = "pretraining_runs/capacity_displacement_5m"
    device: str = "auto"
    seed: int = 20260820
    model_seed: int = 20260820
    context_length: int = 16
    batch_size: int = 256
    fact_batch_fraction: float = 0.5
    max_steps: int = 5_000
    arithmetic_pretrain_steps: int = 0
    arithmetic_pretrain_eval_every: int = 1_000
    reset_optimizer_after_arithmetic_pretrain: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    fact_loss_weight: float = 0.5
    arithmetic_loss_weight: float = 0.5
    log_every: int = 25
    eval_every: int = 1_000
    curve_eval_cases: int = 4_096
    save_checkpoints: bool = True


@dataclass
class DisplacementExperimentConfig:
    model: ToyModelConfig = field(
        default_factory=lambda: ToyModelConfig(
            n_layer=4,
            n_embd=320,
            n_head=4,
            n_inner=1280,
            baseline_n_inner=1298,
        )
    )
    data: DisplacementDataConfig = field(default_factory=DisplacementDataConfig)
    training: DisplacementTrainingConfig = field(default_factory=DisplacementTrainingConfig)


def _merge(dc, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(dc, key):
            raise ValueError(f"Unknown configuration key {key!r} for {type(dc).__name__}")
        current = getattr(dc, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _merge(current, value)
        else:
            setattr(dc, key, value)


def load_displacement_config(path: str | Path) -> DisplacementExperimentConfig:
    config = DisplacementExperimentConfig()
    with open(path) as handle:
        _merge(config, yaml.safe_load(handle) or {})
    validate_config(config)
    return config


def validate_config(config: DisplacementExperimentConfig) -> None:
    if config.training.context_length < 16:
        raise ValueError("context_length must be at least 16 for three-digit arithmetic")
    if config.training.batch_size < 2:
        raise ValueError("batch_size must be at least two")
    fact_batch_size = round(config.training.batch_size * config.training.fact_batch_fraction)
    if not 0 < fact_batch_size < config.training.batch_size:
        raise ValueError("fact_batch_fraction must assign at least one example to each task")
    if config.data.fact_count > config.data.fact_key_vocab_size**2:
        raise ValueError("fact_count exceeds the number of distinct compositional keys")
    if config.data.arithmetic_pool_size < config.data.arithmetic_eval_cases:
        raise ValueError("arithmetic_pool_size must cover all arithmetic evaluation cases")
    weights = config.training.fact_loss_weight + config.training.arithmetic_loss_weight
    if not math.isclose(weights, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("fact_loss_weight and arithmetic_loss_weight must sum to one")


def _toy_config(config: DisplacementExperimentConfig) -> ToyExperimentConfig:
    """Adapt the controlled config to the repository's shared model builder."""
    return ToyExperimentConfig(
        model=config.model,
        data=ToyDataConfig(
            tokenizer_file=config.data.tokenizer_file,
            tokenizer_vocab_size=config.data.tokenizer_vocab_size,
            max_digits=config.data.max_digits,
        ),
        training=ToyTrainingConfig(
            output_dir=config.training.output_dir,
            device=config.training.device,
            seed=config.training.seed,
            model_seed=config.training.model_seed,
            context_length=config.training.context_length,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            max_steps=config.training.max_steps,
            log_every=config.training.log_every,
        ),
    )


def _padded_target_example(
    tokenizer: ArithmeticBPETokenizer,
    text: str,
    target_start: int,
    context_length: int,
) -> tuple[list[int], list[int], list[int]]:
    ids = tokenizer.encode(text)
    if len(ids) > context_length:
        raise ValueError(
            f"Example has {len(ids)} tokens but context_length={context_length}: {text!r}"
        )
    if not 0 < target_start < len(ids):
        raise ValueError("target_start must select at least one non-initial target token")
    padding = context_length - len(ids)
    input_ids = ids + [tokenizer.pad_token_id] * padding
    attention_mask = [1] * len(ids) + [0] * padding
    labels = [-100] * context_length
    labels[target_start:len(ids)] = ids[target_start:]
    return input_ids, attention_mask, labels


def prepare_displacement_data(config: DisplacementExperimentConfig) -> dict[str, object]:
    """Create one deterministic dataset reused by all four conditions."""
    validate_config(config)
    facts = make_compositional_facts(
        config.data.fact_count,
        config.training.seed + 5,
        config.data.fact_key_vocab_size,
        config.data.fact_value_vocab_size,
    )
    atomic_tokens = compositional_fact_special_tokens(
        config.data.fact_key_vocab_size,
        config.data.fact_value_vocab_size,
    )
    tokenizer_texts = [
        f"{fact.key_a}{fact.key_b}<factmap>{fact.value}" for fact in facts[:4096]
    ]
    tokenizer_texts += ["123+45=168", "999-111=888", "21*12=252", "84/7=12"]
    tokenizer = ArithmeticBPETokenizer.train(
        tokenizer_texts,
        config.data.tokenizer_vocab_size,
        atomic_tokens=atomic_tokens,
    )
    tokenizer.save(config.data.tokenizer_file)

    fact_inputs, fact_masks, fact_labels = [], [], []
    for fact in facts:
        prompt = f"{fact.key_a}{fact.key_b}<factmap>"
        prompt_ids = tokenizer.encode(prompt)
        input_ids, mask, labels = _padded_target_example(
            tokenizer,
            prompt + fact.value,
            len(prompt_ids),
            config.training.context_length,
        )
        fact_inputs.append(input_ids)
        fact_masks.append(mask)
        fact_labels.append(labels)

    arithmetic_count = config.data.arithmetic_pool_size
    def arithmetic_examples(
        count: int,
        seed: int,
        excluded: set[str] | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[tuple[str, str]], set[str]]:
        rng = random.Random(seed)
        used = set() if excluded is None else set(excluded)
        inputs, masks, labels, texts = [], [], [], []
        while len(inputs) < count:
            a, op, b, result = _sample_expression(rng, config.data.max_digits)
            prompt = f"{a}{op}{b}="
            answer = str(result)
            identity = prompt + answer
            if identity in used:
                continue
            used.add(identity)
            input_ids, mask, target_labels = _padded_target_example(
                tokenizer,
                identity,
                len(tokenizer.encode(prompt)),
                config.training.context_length,
            )
            inputs.append(input_ids)
            masks.append(mask)
            labels.append(target_labels)
            texts.append((prompt, answer))
        return inputs, masks, labels, texts, used

    (
        arithmetic_inputs,
        arithmetic_masks,
        arithmetic_labels,
        arithmetic_text,
        training_expressions,
    ) = arithmetic_examples(arithmetic_count, config.training.seed + 11)
    (
        arithmetic_eval_inputs,
        arithmetic_eval_masks,
        arithmetic_eval_labels,
        arithmetic_eval_text,
        _,
    ) = arithmetic_examples(
        config.data.arithmetic_eval_cases,
        config.training.seed + 12,
        training_expressions,
    )

    fact_batch_size = round(config.training.batch_size * config.training.fact_batch_fraction)
    arithmetic_batch_size = config.training.batch_size - fact_batch_size
    total_fact_draws = config.training.max_steps * fact_batch_size
    total_arithmetic_draws = config.training.max_steps * arithmetic_batch_size

    def shuffled_schedule(size: int, total: int, seed: int) -> Tensor:
        schedule_rng = torch.Generator().manual_seed(seed)
        schedule = torch.empty(total, dtype=torch.long)
        offset = 0
        while offset < total:
            permutation = torch.randperm(size, generator=schedule_rng)
            take = min(size, total - offset)
            schedule[offset:offset + take] = permutation[:take]
            offset += take
        return schedule

    fact_schedule = shuffled_schedule(
        len(facts), total_fact_draws, config.training.seed + 17
    )
    arithmetic_schedule = shuffled_schedule(
        arithmetic_count, total_arithmetic_draws, config.training.seed + 23
    )

    dataset: dict[str, object] = {
        "fact_input_ids": torch.tensor(fact_inputs, dtype=torch.long),
        "fact_attention_mask": torch.tensor(fact_masks, dtype=torch.long),
        "fact_labels": torch.tensor(fact_labels, dtype=torch.long),
        "arithmetic_input_ids": torch.tensor(arithmetic_inputs, dtype=torch.long),
        "arithmetic_attention_mask": torch.tensor(arithmetic_masks, dtype=torch.long),
        "arithmetic_labels": torch.tensor(arithmetic_labels, dtype=torch.long),
        "arithmetic_eval_input_ids": torch.tensor(arithmetic_eval_inputs, dtype=torch.long),
        "arithmetic_eval_attention_mask": torch.tensor(arithmetic_eval_masks, dtype=torch.long),
        "arithmetic_eval_labels": torch.tensor(arithmetic_eval_labels, dtype=torch.long),
        "fact_schedule": fact_schedule.view(config.training.max_steps, fact_batch_size),
        "arithmetic_schedule": arithmetic_schedule.view(config.training.max_steps, arithmetic_batch_size),
        "arithmetic_training_text": arithmetic_text,
        "arithmetic_eval_text": arithmetic_eval_text,
        "fact_eval_cases": compositional_fact_eval_cases(facts),
        "metadata": {
            "protocol": "target_masked_capacity_displacement_v1",
            "fact_count": len(facts),
            "fact_value_classes": config.data.fact_value_vocab_size,
            "fact_information_bits": len(facts) * math.log2(config.data.fact_value_vocab_size),
            "fact_draws_per_condition": total_fact_draws,
            "mean_fact_exposures": total_fact_draws / len(facts),
            "arithmetic_draws_per_mixed_condition": total_arithmetic_draws,
            "arithmetic_pretrain_draws_per_mixed_condition": (
                config.training.arithmetic_pretrain_steps * config.training.batch_size
            ),
            "mean_arithmetic_pretrain_exposures": (
                config.training.arithmetic_pretrain_steps
                * config.training.batch_size
                / arithmetic_count
            ),
            "arithmetic_pool_size": arithmetic_count,
            "arithmetic_eval_cases": config.data.arithmetic_eval_cases,
            "arithmetic_eval_is_disjoint": True,
            "context_length": config.training.context_length,
            "batch_size": config.training.batch_size,
            "fact_batch_fraction": config.training.fact_batch_fraction,
            "loss": (
                f"{config.training.fact_loss_weight:g} normalized fact target loss + "
                f"{config.training.arithmetic_loss_weight:g} normalized arithmetic answer loss"
            ),
            "seed": config.training.seed,
        },
    }
    path = Path(config.data.dataset_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, path)
    return dataset


def _load_data(config: DisplacementExperimentConfig, prepare: bool) -> dict[str, object]:
    path = Path(config.data.dataset_file)
    if prepare or not path.exists() or not Path(config.data.tokenizer_file).exists():
        return prepare_displacement_data(config)
    return torch.load(path, map_location="cpu", weights_only=False)


def _target_loss(logits: Tensor, labels: Tensor) -> Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _forward_logits(model: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs["logits"] if isinstance(outputs, dict) else outputs.logits


def _batch_loss(model: nn.Module, batch: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    input_ids, attention_mask, labels = batch
    return _target_loss(_forward_logits(model, input_ids, attention_mask), labels)


def _zero_compute_hook(_module, _args, output):
    return replace(output, results=torch.zeros_like(output.results))


def _install_zero_compute_control(model: nn.Module) -> None:
    if not isinstance(model, GPT2WithARB):
        raise TypeError("zero-compute control requires GPT2WithARB")
    handle = model.compute_core.register_forward_hook(_zero_compute_hook)
    # Keep the hook alive and make the control explicit in checkpoints.
    model._capacity_displacement_zero_hook = handle


def _select_batch(data: dict[str, object], prefix: str, indices: Tensor, device: torch.device):
    return tuple(
        data[f"{prefix}_{suffix}"][indices].to(device)
        for suffix in ("input_ids", "attention_mask", "labels")
    )


def recoverable_fact_bits(accuracy: float, count: int, classes: int) -> float:
    """Fano-style lower bound on information recovered from balanced labels."""
    chance = 1.0 / classes
    if count <= 0 or classes <= 1 or accuracy <= chance:
        return 0.0
    error = min(max(1.0 - accuracy, 0.0), 1.0)
    binary_entropy = 0.0
    if 0.0 < error < 1.0:
        binary_entropy = -error * math.log2(error) - (1.0 - error) * math.log2(1.0 - error)
    bits_per_fact = math.log2(classes) - binary_entropy - error * math.log2(classes - 1)
    return max(0.0, bits_per_fact) * count


def _evaluate_target_batches(
    model: nn.Module,
    data: dict[str, object],
    prefix: str,
    device: torch.device,
    batch_size: int,
    max_cases: int | None = None,
) -> dict[str, float]:
    inputs: Tensor = data[f"{prefix}_input_ids"]
    masks: Tensor = data[f"{prefix}_attention_mask"]
    labels: Tensor = data[f"{prefix}_labels"]
    if max_cases is not None:
        inputs = inputs[:max_cases]
        masks = masks[:max_cases]
        labels = labels[:max_cases]
    total_loss = 0.0
    target_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[start:start + batch_size].to(device)
            batch_masks = masks[start:start + batch_size].to(device)
            batch_labels = labels[start:start + batch_size].to(device)
            logits = _forward_logits(model, batch_inputs, batch_masks)
            shifted_logits = logits[:, :-1, :]
            shifted_labels = batch_labels[:, 1:]
            valid = shifted_labels.ne(-100)
            losses = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shifted_labels)
            predictions = shifted_logits.argmax(dim=-1)
            matches = predictions.eq(shifted_labels) | ~valid
            total_loss += float((losses * valid).sum().item())
            target_tokens += int(valid.sum().item())
            correct_tokens += int((predictions.eq(shifted_labels) & valid).sum().item())
            correct_sequences += int(matches.all(dim=1).sum().item())
    return {
        "target_nll": total_loss / max(target_tokens, 1),
        "target_token_accuracy": correct_tokens / max(target_tokens, 1),
        "exact_sequence_accuracy": correct_sequences / len(inputs),
        "eval_cases": len(inputs),
    }


def _evaluate_arithmetic_by_operation(
    model: nn.Module,
    data: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    prompts = [prompt for prompt, _answer in data["arithmetic_eval_text"]]
    results: dict[str, float] = {}
    for operation in ("+", "-", "*", "/"):
        indices = torch.tensor(
            [index for index, prompt in enumerate(prompts) if operation in prompt],
            dtype=torch.long,
        )
        name = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}[operation]
        if not len(indices):
            results[f"arithmetic_{name}_exact_accuracy"] = float("nan")
            results[f"arithmetic_{name}_eval_cases"] = 0
            continue
        subset = {
            "op_input_ids": data["arithmetic_eval_input_ids"][indices],
            "op_attention_mask": data["arithmetic_eval_attention_mask"][indices],
            "op_labels": data["arithmetic_eval_labels"][indices],
        }
        metrics = _evaluate_target_batches(
            model, subset, "op", device, batch_size
        )
        results[f"arithmetic_{name}_exact_accuracy"] = metrics["exact_sequence_accuracy"]
        results[f"arithmetic_{name}_eval_cases"] = metrics["eval_cases"]
    return results


def _shared_body_parameters(model: nn.Module) -> list[nn.Parameter]:
    base = model.base_model if isinstance(model, GPT2WithARB) else model
    return [
        parameter for name, parameter in base.named_parameters()
        if name.startswith("transformer.") and parameter.requires_grad
    ]


def _gradient_vector(model: nn.Module, loss: Tensor) -> tuple[list[Tensor], float]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradients = []
    squared_norm = 0.0
    for parameter in _shared_body_parameters(model):
        gradient = parameter.grad
        clone = torch.zeros_like(parameter, memory_format=torch.preserve_format) if gradient is None else gradient.detach().clone()
        gradients.append(clone)
        squared_norm += float(clone.float().square().sum().item())
    return gradients, math.sqrt(squared_norm)


def _gradient_diagnostics(
    model: nn.Module,
    fact_batch,
    arithmetic_batch,
) -> dict[str, float]:
    model.train()
    fact_gradients, fact_norm = _gradient_vector(model, _batch_loss(model, fact_batch))
    arithmetic_gradients, arithmetic_norm = _gradient_vector(
        model, _batch_loss(model, arithmetic_batch)
    )
    dot = sum(
        float((fact.float() * arithmetic.float()).sum().item())
        for fact, arithmetic in zip(fact_gradients, arithmetic_gradients)
    )
    cosine = dot / (fact_norm * arithmetic_norm) if fact_norm and arithmetic_norm else 0.0
    model.zero_grad(set_to_none=True)
    return {
        "shared_body_fact_gradient_norm": fact_norm,
        "shared_body_arithmetic_gradient_norm": arithmetic_norm,
        "shared_body_fact_arithmetic_gradient_cosine": cosine,
    }


def _condition_model(
    config: DisplacementExperimentConfig,
    condition: str,
    tokenizer: ArithmeticBPETokenizer,
) -> nn.Module:
    variant = "arb" if condition in {"arb_mixed", "zero_arb_mixed"} else "baseline"
    model = build_model(_toy_config(config), variant, tokenizer)
    if condition == "zero_arb_mixed":
        _install_zero_compute_control(model)
    return model


def run_condition(
    config: DisplacementExperimentConfig,
    condition: str,
    data: dict[str, object],
) -> dict[str, object]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    tokenizer = ArithmeticBPETokenizer.from_file(config.data.tokenizer_file)
    device = resolve_device(config.training.device)
    model = _condition_model(config, condition, tokenizer).to(device)
    torch.manual_seed(config.training.seed)
    random.seed(config.training.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    fact_batch_size = data["fact_schedule"].shape[1]
    arithmetic_batch_size = data["arithmetic_schedule"].shape[1]
    history: list[dict[str, float]] = []
    evaluation_history: list[dict[str, float]] = []
    arithmetic_pretrain_history: list[dict[str, float]] = []

    if condition != "fact_only" and config.training.arithmetic_pretrain_steps:
        pretrain_total = config.training.arithmetic_pretrain_steps * config.training.batch_size
        pretrain_generator = torch.Generator().manual_seed(config.training.seed + 29)
        pretrain_schedule = torch.empty(pretrain_total, dtype=torch.long)
        offset = 0
        while offset < pretrain_total:
            permutation = torch.randperm(len(data["arithmetic_input_ids"]), generator=pretrain_generator)
            take = min(len(permutation), pretrain_total - offset)
            pretrain_schedule[offset:offset + take] = permutation[:take]
            offset += take
        pretrain_schedule = pretrain_schedule.view(
            config.training.arithmetic_pretrain_steps,
            config.training.batch_size,
        )
        model.train()
        pretrain_progress = tqdm(
            range(config.training.arithmetic_pretrain_steps),
            desc=f"{condition} arithmetic pretrain ({device.type})",
            unit="step",
            dynamic_ncols=True,
        )
        for step in pretrain_progress:
            arithmetic_batch = _select_batch(
                data, "arithmetic", pretrain_schedule[step], device
            )
            loss = _batch_loss(model, arithmetic_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 0 or (step + 1) % config.training.log_every == 0 or step + 1 == config.training.arithmetic_pretrain_steps:
                pretrain_progress.set_postfix(math=f"{float(loss.item()):.3f}")
            if (
                config.training.arithmetic_pretrain_eval_every
                and (step + 1) % config.training.arithmetic_pretrain_eval_every == 0
            ):
                arithmetic_curve = _evaluate_target_batches(
                    model,
                    data,
                    "arithmetic_eval",
                    device,
                    config.training.batch_size,
                )
                arithmetic_pretrain_history.append({
                    "step": step + 1,
                    "arithmetic_target_nll": arithmetic_curve["target_nll"],
                    "arithmetic_exact_accuracy": arithmetic_curve["exact_sequence_accuracy"],
                    **_evaluate_arithmetic_by_operation(
                        model, data, device, config.training.batch_size
                    ),
                })
                model.train()
        if config.training.reset_optimizer_after_arithmetic_pretrain:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )

    model.train()
    progress = tqdm(
        range(config.training.max_steps),
        desc=f"{condition} ({device.type})",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:
        fact_indices = data["fact_schedule"][step]
        fact_batch = _select_batch(data, "fact", fact_indices, device)
        fact_loss = _batch_loss(model, fact_batch)
        if condition == "fact_only":
            arithmetic_loss = None
            # Match the fact-gradient coefficient in the mixed conditions.
            # Otherwise the fact-only reference receives twice the effective
            # factual learning rate and cannot isolate task interference.
            loss = config.training.fact_loss_weight * fact_loss
        else:
            arithmetic_indices = data["arithmetic_schedule"][step]
            arithmetic_batch = _select_batch(data, "arithmetic", arithmetic_indices, device)
            arithmetic_loss = _batch_loss(model, arithmetic_batch)
            loss = (
                config.training.fact_loss_weight * fact_loss
                + config.training.arithmetic_loss_weight * arithmetic_loss
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % config.training.log_every == 0 or step + 1 == config.training.max_steps:
            record = {
                "step": step + 1,
                "total_loss": float(loss.item()),
                "fact_loss": float(fact_loss.item()),
                "arithmetic_loss": float(arithmetic_loss.item()) if arithmetic_loss is not None else float("nan"),
            }
            history.append(record)
            progress.set_postfix(
                fact=f"{record['fact_loss']:.3f}",
                math="-" if arithmetic_loss is None else f"{record['arithmetic_loss']:.3f}",
            )
        if config.training.eval_every and (step + 1) % config.training.eval_every == 0:
            fact_curve = _evaluate_target_batches(
                model,
                data,
                "fact",
                device,
                config.training.batch_size,
                max_cases=config.training.curve_eval_cases,
            )
            arithmetic_curve = _evaluate_target_batches(
                model,
                data,
                "arithmetic_eval",
                device,
                config.training.batch_size,
            )
            evaluation_history.append({
                "step": step + 1,
                "fact_target_nll": fact_curve["target_nll"],
                "fact_accuracy": fact_curve["exact_sequence_accuracy"],
                "arithmetic_target_nll": arithmetic_curve["target_nll"],
                "arithmetic_exact_accuracy": arithmetic_curve["exact_sequence_accuracy"],
            })
            model.train()

    fact_metrics = _evaluate_target_batches(
        model, data, "fact", device, config.training.batch_size
    )
    arithmetic_metrics = _evaluate_target_batches(
        model, data, "arithmetic_eval", device, config.training.batch_size
    )
    diagnostic_indices = data["fact_schedule"][0]
    fact_batch = _select_batch(data, "fact", diagnostic_indices, device)
    arithmetic_batch = _select_batch(
        data, "arithmetic", torch.arange(arithmetic_batch_size), device
    )
    gradients = _gradient_diagnostics(model, fact_batch, arithmetic_batch)
    fact_accuracy = fact_metrics["exact_sequence_accuracy"]
    recovered_bits = recoverable_fact_bits(
        fact_accuracy,
        config.data.fact_count,
        config.data.fact_value_vocab_size,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics: dict[str, object] = {
        "condition": condition,
        "parameter_count": parameter_count,
        "device": str(device),
        "training_history": history,
        "evaluation_history": evaluation_history,
        "arithmetic_pretrain_history": arithmetic_pretrain_history,
        "fact_target_nll": fact_metrics["target_nll"],
        "fact_accuracy": fact_accuracy,
        "fact_eval_cases": fact_metrics["eval_cases"],
        "recoverable_fact_bits_lower_bound": recovered_bits,
        "recoverable_fact_bits_per_parameter": recovered_bits / parameter_count,
        "arithmetic_target_nll": arithmetic_metrics["target_nll"],
        "arithmetic_digit_accuracy": arithmetic_metrics["target_token_accuracy"],
        "arithmetic_exact_accuracy": arithmetic_metrics["exact_sequence_accuracy"],
        "arithmetic_eval_cases": arithmetic_metrics["eval_cases"],
        **_evaluate_arithmetic_by_operation(
            model, data, device, config.training.batch_size
        ),
        **gradients,
    }
    output_dir = Path(config.training.output_dir) / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if config.training.save_checkpoints:
        torch.save(
            {"model_state": model.state_dict(), "config": asdict(config), "metrics": metrics},
            output_dir / "checkpoint.pt",
        )
    return metrics


def run_experiment(
    config: DisplacementExperimentConfig,
    *,
    prepare: bool = False,
    conditions: Iterable[str] = CONDITIONS,
) -> dict[str, object]:
    data = _load_data(config, prepare)
    results = {
        condition: run_condition(config, condition, data)
        for condition in conditions
    }
    summary = {
        "config": asdict(config),
        "data_metadata": data["metadata"],
        "results": results,
    }
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
