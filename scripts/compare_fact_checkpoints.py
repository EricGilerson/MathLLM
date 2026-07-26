#!/usr/bin/env python3
"""Paired exact-recall diagnostic for saved toy baseline/ARB checkpoints.

Reports which fixed fact bindings are correct for both models, baseline only,
ARB only, or neither.  This is an evaluation-only companion to the
compositional capacity probe; it performs no training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathllm.pretraining.arithmetic_bpe_tokenizer import ArithmeticBPETokenizer
from mathllm.pretraining.experiment import build_model, load_toy_config, resolve_device
from mathllm.pretraining.fact_benchmark import (
    atomic_fact_eval_cases,
    compositional_fact_eval_cases,
    make_atomic_facts,
    make_compositional_facts,
)


def _cases(config):
    if config.data.fact_format == "compositional":
        facts = make_compositional_facts(
            config.data.fact_count, config.training.seed + 5,
            config.data.fact_key_vocab_size, config.data.fact_value_vocab_size,
        )
        return compositional_fact_eval_cases(facts)
    if config.data.fact_format == "atomic":
        return atomic_fact_eval_cases(make_atomic_facts(config.data.fact_count, config.training.seed + 5))
    raise ValueError("This diagnostic requires fact_format='atomic' or 'compositional'")


def _predictions(config, tokenizer, variant: str, cases, device: torch.device, batch_size: int):
    model = build_model(config, variant, tokenizer).to(device).eval()
    checkpoint = Path(config.training.output_dir) / variant / "checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
    prompts, expected = zip(*cases)
    encoded = [tokenizer.encode(prompt) for prompt in prompts]
    targets = [tokenizer.encode(value) for value in expected]
    if len({len(ids) for ids in encoded}) != 1 or any(len(ids) != 1 for ids in targets):
        raise ValueError("Expected fixed-length prompts and one-token values")
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            inputs = torch.tensor(encoded[start:start + batch_size], dtype=torch.long, device=device)
            answer_ids = torch.tensor(targets[start:start + batch_size], dtype=torch.long, device=device)[:, 0]
            outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            pieces.append((logits[:, -1, :].argmax(dim=-1) == answer_ids).cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    config = load_toy_config(args.config)
    device = resolve_device(args.device)
    tokenizer = ArithmeticBPETokenizer.from_file(config.data.tokenizer_file)
    cases = _cases(config)
    baseline = _predictions(config, tokenizer, "baseline", cases, device, args.batch_size)
    arb = _predictions(config, tokenizer, "arb", cases, device, args.batch_size)
    both_correct = int((baseline & arb).sum())
    baseline_only = int((baseline & ~arb).sum())
    arb_only = int((~baseline & arb).sum())
    both_wrong = int((~baseline & ~arb).sum())
    discordant = baseline_only + arb_only
    # Continuity-corrected McNemar statistic; the contingency counts, not a
    # single-model binomial approximation, are the important diagnostic.
    mcnemar = ((abs(arb_only - baseline_only) - 1) ** 2 / discordant) if discordant else 0.0
    print(json.dumps({
        "fact_eval_cases": len(cases),
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "arb_only_correct": arb_only,
        "both_wrong": both_wrong,
        "net_arb_correct_bindings": arb_only - baseline_only,
        "mcnemar_chi2_continuity_corrected": mcnemar,
    }, indent=2))


if __name__ == "__main__":
    main()
