"""Language-retention benchmark for frozen base vs. ARB-augmented models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlretrieve
from zipfile import ZipFile

import torch
import torch.nn.functional as F
from datasets import load_dataset

logger = logging.getLogger(__name__)

PIQA_TRAIN_DEV_URL = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
PIQA_CACHE_DIR = Path.home() / ".cache" / "mathllm" / "piqa"


@dataclass(frozen=True)
class MultipleChoiceExample:
    """One multiple-choice example scored by continuation likelihood."""

    prompt: str
    choices: tuple[str, ...]
    label: int
    source_id: str = ""


def _prepare_model(model, device: torch.device) -> None:
    """Move the model once and warm any runtime caches."""
    model.eval()
    model.to(device)
    if hasattr(model, "prepare_for_device"):
        model.prepare_for_device(device)


def _extract_loss(outputs):
    """Extract loss from either a dict-like or HF model output."""
    if isinstance(outputs, dict):
        return outputs["loss"]
    return outputs.loss


def _extract_logits(outputs):
    """Extract logits from either a dict-like or HF model output."""
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs.logits


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace for prompt formatting."""
    return " ".join(str(text).split())


def _resolve_split(split: str, limit: int | None) -> str:
    """Convert a split name plus limit into a datasets-compatible slice."""
    if limit is None or limit <= 0:
        return split
    return f"{split}[:{limit}]"


def _resolve_pad_token_id(tokenizer, model) -> int:
    """Find a pad token for batched scoring."""
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    model_cfg = getattr(model, "config", None)
    if model_cfg is not None and getattr(model_cfg, "eos_token_id", None) is not None:
        return int(model_cfg.eos_token_id)
    raise ValueError("Tokenizer or model must define pad_token_id or eos_token_id")


def load_wikitext_103_texts(
    limit: int | None = 256,
    *,
    split: str = "test",
    dataset_loader: Callable[..., Any] = load_dataset,
) -> list[str]:
    """Load non-empty WikiText-103 raw text samples."""
    rows = dataset_loader(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=_resolve_split(split, limit),
    )
    texts = [_normalize_whitespace(row["text"]) for row in rows]
    return [text for text in texts if text]


def load_piqa_examples(
    limit: int | None = 512,
    *,
    split: str = "validation",
    dataset_loader: Callable[..., Any] = load_dataset,
) -> list[MultipleChoiceExample]:
    """Load PIQA as prompt + two answer continuations."""
    try:
        rows = dataset_loader("piqa", split=_resolve_split(split, limit))
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        logger.warning(
            "datasets rejected the legacy PIQA script; falling back to the original source files"
        )
        return _load_piqa_examples_from_original_files(limit=limit, split=split)

    examples = []
    for idx, row in enumerate(rows):
        examples.append(
            MultipleChoiceExample(
                prompt=f"Question: {_normalize_whitespace(row['goal'])}\nAnswer:",
                choices=(
                    f" {_normalize_whitespace(row['sol1'])}",
                    f" {_normalize_whitespace(row['sol2'])}",
                ),
                label=int(row["label"]),
                source_id=str(row.get("id", idx)),
            )
        )
    return examples


def _download_if_missing(url: str, destination: Path) -> Path:
    """Download a file into the local benchmark cache if needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(url, destination)
    return destination


def _load_piqa_examples_from_original_files(
    limit: int | None,
    *,
    split: str = "validation",
    cache_dir: Path = PIQA_CACHE_DIR,
) -> list[MultipleChoiceExample]:
    """Load PIQA directly from the original published files.

    Newer `datasets` releases reject legacy script-backed loaders. PIQA's old
    Hugging Face loader points to the original AI2/Mosaic zip, so we mirror
    that logic here for the validation benchmark.
    """
    if split != "validation":
        raise ValueError("PIQA fallback currently supports only the validation split")

    archive_path = _download_if_missing(
        PIQA_TRAIN_DEV_URL,
        cache_dir / "physicaliqa-train-dev.zip",
    )
    extracted_root = cache_dir / "physicaliqa-train-dev"
    input_path = extracted_root / "dev.jsonl"
    label_path = extracted_root / "dev-labels.lst"

    if not input_path.exists() or not label_path.exists():
        with ZipFile(archive_path) as zip_file:
            zip_file.extractall(cache_dir)

    with open(input_path, encoding="utf-8") as input_file:
        rows = input_file.read().splitlines()
    with open(label_path, encoding="utf-8") as label_file:
        labels = label_file.read().splitlines()

    if limit is not None and limit > 0:
        rows = rows[:limit]
        labels = labels[:limit]

    examples = []
    for idx, (row, label) in enumerate(zip(rows, labels)):
        record = json.loads(row)
        examples.append(
            MultipleChoiceExample(
                prompt=f"Question: {_normalize_whitespace(record['goal'])}\nAnswer:",
                choices=(
                    f" {_normalize_whitespace(record['sol1'])}",
                    f" {_normalize_whitespace(record['sol2'])}",
                ),
                label=int(label),
                source_id=str(idx),
            )
        )
    return examples


def load_hellaswag_examples(
    limit: int | None = 512,
    *,
    split: str = "validation",
    dataset_loader: Callable[..., Any] = load_dataset,
) -> list[MultipleChoiceExample]:
    """Load HellaSwag as context + four candidate endings."""
    rows = dataset_loader("hellaswag", split=_resolve_split(split, limit))
    examples = []
    for idx, row in enumerate(rows):
        context = row.get("ctx")
        if not context:
            ctx_a = _normalize_whitespace(row.get("ctx_a", ""))
            ctx_b = _normalize_whitespace(row.get("ctx_b", ""))
            context = f"{ctx_a} {ctx_b}".strip()

        activity = _normalize_whitespace(row.get("activity_label", ""))
        if activity:
            context = f"{activity}: {context}"

        endings = tuple(f" {_normalize_whitespace(choice)}" for choice in row["endings"])
        examples.append(
            MultipleChoiceExample(
                prompt=f"Context: {_normalize_whitespace(context)}\nEnding:",
                choices=endings,
                label=int(row["label"]),
                source_id=str(row.get("ind", idx)),
            )
        )
    return examples


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int = 512,
    stride: int = 512,
) -> dict[str, float]:
    """Compute token-stream perplexity over concatenated texts."""
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1")
    if stride <= 0:
        raise ValueError("stride must be positive")

    _prepare_model(model, device)

    separator_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    token_ids: list[int] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        if token_ids and separator_ids:
            token_ids.extend(separator_ids)
        token_ids.extend(encoded)

    if len(token_ids) < 2:
        raise ValueError("Need at least two tokens to compute perplexity")

    all_ids = torch.tensor(token_ids, dtype=torch.long)
    total_nll = 0.0
    total_tokens = 0
    prev_end_loc = 0

    for begin_step in range(0, all_ids.size(0), stride):
        end_loc = min(begin_step + stride, all_ids.size(0))
        begin_loc = max(end_loc - max_length, 0)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0:
            continue

        input_ids = all_ids[begin_loc:end_loc].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :-trg_len] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = _extract_loss(outputs)
        total_nll += float(loss.item()) * trg_len
        total_tokens += trg_len
        prev_end_loc = end_loc
        if end_loc >= all_ids.size(0):
            break

    avg_nll = total_nll / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    return {
        "avg_nll": avg_nll,
        "perplexity": perplexity,
        "num_tokens": total_tokens,
    }


@torch.no_grad()
def score_continuations(
    model,
    tokenizer,
    prompts: list[str],
    continuations: list[str],
    *,
    device: torch.device,
    batch_size: int = 8,
) -> list[dict[str, float]]:
    """Score continuation log-likelihoods for prompt/continuation pairs."""
    if len(prompts) != len(continuations):
        raise ValueError("prompts and continuations must have the same length")

    _prepare_model(model, device)
    pad_token_id = _resolve_pad_token_id(tokenizer, model)

    records = []
    for prompt, continuation in zip(prompts, continuations):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
        if not continuation_ids:
            raise ValueError(f"Continuation tokenized to zero tokens: {continuation!r}")
        full_ids = prompt_ids + continuation_ids
        labels = [-100] * len(prompt_ids) + continuation_ids
        records.append(
            {
                "input_ids": full_ids,
                "labels": labels,
                "num_choice_tokens": len(continuation_ids),
            }
        )

    scores: list[dict[str, float]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        max_len = max(len(record["input_ids"]) for record in batch)

        input_ids = torch.full(
            (len(batch), max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(batch), max_len),
            dtype=torch.long,
            device=device,
        )
        labels = torch.full(
            (len(batch), max_len),
            -100,
            dtype=torch.long,
            device=device,
        )

        for row, record in enumerate(batch):
            seq_len = len(record["input_ids"])
            input_ids[row, :seq_len] = torch.tensor(record["input_ids"], dtype=torch.long, device=device)
            attention_mask[row, :seq_len] = 1
            labels[row, :seq_len] = torch.tensor(record["labels"], dtype=torch.long, device=device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = _extract_logits(outputs)
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)
        gathered = F.log_softmax(shift_logits, dim=-1).gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        gathered = gathered * valid_mask

        batch_sum_logprobs = gathered.sum(dim=1)
        batch_token_counts = valid_mask.sum(dim=1)
        batch_mean_logprobs = batch_sum_logprobs / batch_token_counts.clamp_min(1)

        for row, record in enumerate(batch):
            scores.append(
                {
                    "sum_logprob": float(batch_sum_logprobs[row].item()),
                    "mean_logprob": float(batch_mean_logprobs[row].item()),
                    "num_choice_tokens": int(record["num_choice_tokens"]),
                }
            )

    return scores


def benchmark_multiple_choice(
    model,
    tokenizer,
    examples: list[MultipleChoiceExample],
    *,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, float]:
    """Benchmark continuation likelihood accuracy on a multiple-choice task."""
    if not examples:
        raise ValueError("Need at least one multiple-choice example")

    prompts: list[str] = []
    continuations: list[str] = []
    choice_counts: list[int] = []

    for example in examples:
        choice_counts.append(len(example.choices))
        prompts.extend([example.prompt] * len(example.choices))
        continuations.extend(example.choices)

    flat_scores = score_continuations(
        model,
        tokenizer,
        prompts,
        continuations,
        device=device,
        batch_size=batch_size,
    )

    accuracy = 0
    normalized_accuracy = 0
    offset = 0
    for example, choice_count in zip(examples, choice_counts):
        choice_scores = flat_scores[offset:offset + choice_count]
        offset += choice_count

        predicted = max(range(choice_count), key=lambda idx: choice_scores[idx]["sum_logprob"])
        normalized_predicted = max(
            range(choice_count),
            key=lambda idx: choice_scores[idx]["mean_logprob"],
        )
        if predicted == example.label:
            accuracy += 1
        if normalized_predicted == example.label:
            normalized_accuracy += 1

    total = len(examples)
    return {
        "accuracy": accuracy / total,
        "length_normalized_accuracy": normalized_accuracy / total,
        "num_examples": total,
    }


def _metric_delta(base_value: float, arb_value: float) -> dict[str, float]:
    """Compute absolute and relative deltas."""
    absolute = arb_value - base_value
    relative = absolute / base_value if base_value != 0 else 0.0
    return {"absolute": absolute, "relative": relative}


def _format_metric_value(metric_name: str, value: float) -> str:
    """Format one metric value for the Markdown table."""
    if metric_name == "perplexity":
        return f"{value:.3f}"
    return f"{value * 100:.2f}%"


def _format_delta(metric_name: str, value: float) -> str:
    """Format one delta value for the Markdown table."""
    if metric_name == "perplexity":
        return f"{value:+.3f}"
    return f"{value * 100:+.2f} pts"


def render_markdown_table(results: dict[str, Any]) -> str:
    """Render a compact paper-ready comparison table."""
    rows = [
        "| Benchmark | Metric | Base | ARB | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for benchmark_name, benchmark in results["benchmarks"].items():
        metric_name = benchmark["primary_metric"]
        base_value = benchmark["base"][metric_name]
        arb_value = benchmark["arb"][metric_name]
        delta_value = benchmark["delta"][metric_name]["absolute"]
        rows.append(
            "| "
            f"{benchmark['display_name']} | {metric_name} | "
            f"{_format_metric_value(metric_name, base_value)} | "
            f"{_format_metric_value(metric_name, arb_value)} | "
            f"{_format_delta(metric_name, delta_value)} |"
        )

    return "\n".join(rows)


def run_zero_forgetting_benchmark(
    *,
    base_model,
    base_tokenizer,
    arb_model,
    arb_tokenizer,
    device: torch.device,
    batch_size: int = 8,
    perplexity_max_length: int = 512,
    perplexity_stride: int = 512,
    wikitext_texts: list[str] | None = None,
    piqa_examples: list[MultipleChoiceExample] | None = None,
    hellaswag_examples: list[MultipleChoiceExample] | None = None,
    wikitext_limit: int = 256,
    piqa_limit: int = 512,
    hellaswag_limit: int = 512,
    dataset_loader: Callable[..., Any] = load_dataset,
) -> dict[str, Any]:
    """Run the frozen-base vs. ARB language-retention comparison."""
    if wikitext_texts is None and wikitext_limit > 0:
        wikitext_texts = load_wikitext_103_texts(
            wikitext_limit,
            dataset_loader=dataset_loader,
        )
    if piqa_examples is None and piqa_limit > 0:
        piqa_examples = load_piqa_examples(
            piqa_limit,
            dataset_loader=dataset_loader,
        )
    if hellaswag_examples is None and hellaswag_limit > 0:
        hellaswag_examples = load_hellaswag_examples(
            hellaswag_limit,
            dataset_loader=dataset_loader,
        )

    benchmarks: dict[str, Any] = {}

    if wikitext_texts:
        base_metrics = compute_perplexity(
            base_model,
            base_tokenizer,
            wikitext_texts,
            device=device,
            max_length=perplexity_max_length,
            stride=perplexity_stride,
        )
        arb_metrics = compute_perplexity(
            arb_model,
            arb_tokenizer,
            wikitext_texts,
            device=device,
            max_length=perplexity_max_length,
            stride=perplexity_stride,
        )
        benchmarks["wikitext_103"] = {
            "display_name": "WikiText-103",
            "primary_metric": "perplexity",
            "num_examples": len(wikitext_texts),
            "base": base_metrics,
            "arb": arb_metrics,
            "delta": {
                "perplexity": _metric_delta(base_metrics["perplexity"], arb_metrics["perplexity"]),
                "avg_nll": _metric_delta(base_metrics["avg_nll"], arb_metrics["avg_nll"]),
            },
        }

    if piqa_examples:
        base_metrics = benchmark_multiple_choice(
            base_model,
            base_tokenizer,
            piqa_examples,
            device=device,
            batch_size=batch_size,
        )
        arb_metrics = benchmark_multiple_choice(
            arb_model,
            arb_tokenizer,
            piqa_examples,
            device=device,
            batch_size=batch_size,
        )
        benchmarks["piqa"] = {
            "display_name": "PIQA",
            "primary_metric": "accuracy",
            "num_examples": len(piqa_examples),
            "base": base_metrics,
            "arb": arb_metrics,
            "delta": {
                "accuracy": _metric_delta(base_metrics["accuracy"], arb_metrics["accuracy"]),
                "length_normalized_accuracy": _metric_delta(
                    base_metrics["length_normalized_accuracy"],
                    arb_metrics["length_normalized_accuracy"],
                ),
            },
        }

    if hellaswag_examples:
        base_metrics = benchmark_multiple_choice(
            base_model,
            base_tokenizer,
            hellaswag_examples,
            device=device,
            batch_size=batch_size,
        )
        arb_metrics = benchmark_multiple_choice(
            arb_model,
            arb_tokenizer,
            hellaswag_examples,
            device=device,
            batch_size=batch_size,
        )
        benchmarks["hellaswag"] = {
            "display_name": "HellaSwag",
            "primary_metric": "accuracy",
            "num_examples": len(hellaswag_examples),
            "base": base_metrics,
            "arb": arb_metrics,
            "delta": {
                "accuracy": _metric_delta(base_metrics["accuracy"], arb_metrics["accuracy"]),
                "length_normalized_accuracy": _metric_delta(
                    base_metrics["length_normalized_accuracy"],
                    arb_metrics["length_normalized_accuracy"],
                ),
            },
        }

    results = {"benchmarks": benchmarks}
    results["markdown_table"] = render_markdown_table(results)
    return results
