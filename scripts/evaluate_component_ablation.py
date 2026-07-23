#!/usr/bin/env python3
"""Run inference-time ARB injection and LoRA dependency ablations.

These settings use one trained checkpoint. They test which existing components
the result depends on; they do not test whether a smaller LoRA rank can train.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import torch

from mathllm.evaluation.evaluator import ARBEvaluator
from mathllm.model.gpt2_arb import GPT2WithARB


def _set_multipliers(model: GPT2WithARB, injection: float, lora: float) -> None:
    for injector in model.injectors.values():
        injector.inject.set_eval_gate_multiplier(injection)
    if model.lora_head is not None:
        model.lora_head.set_eval_multiplier(lora)
    if model.lora_layers is not None:
        for layer in model.lora_layers.values():
            layer.set_eval_multiplier(lora)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="trained_model_360m")
    parser.add_argument("--output", default="review/artifacts/component_ablation.json")
    parser.add_argument("--samples-per-cell", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--setting", choices=["full", "injection_off", "lora_off", "both_off"],
                        help="Run one setting only, for execution-window sharding.")
    args = parser.parse_args()
    if args.samples_per_cell < 1:
        parser.error("--samples-per-cell must be positive")

    device = torch.device(args.device)
    model, tokenizer, config = GPT2WithARB.from_exported_model(args.model_dir, device=device)
    config.evaluation.batch_size = args.batch_size
    evaluator = ARBEvaluator(model, tokenizer, config.evaluation, device)
    cells = evaluator._build_exact_match_cells(args.samples_per_cell, args.seed)
    cells += evaluator._build_division_cells(args.samples_per_cell, args.seed + 1)
    cases = [case for cell in cells for case in cell["cases"]]

    settings = {
        "full": (1.0, 1.0),
        "injection_off": (0.0, 1.0),
        "lora_off": (1.0, 0.0),
        "both_off": (0.0, 0.0),
    }
    results = {}
    selected_settings = {args.setting: settings[args.setting]} if args.setting else settings
    for name, (injection, lora) in selected_settings.items():
        _set_multipliers(model, injection, lora)
        score = evaluator._evaluate_integer_cases(cases, include_examples=True)
        results[name] = {
            "injection_multiplier": injection,
            "lora_multiplier": lora,
            **score,
        }

    report = {
        "date": str(date.today()), "model_dir": args.model_dir, "device": str(device),
        "seed": args.seed, "samples_per_cell": args.samples_per_cell,
        "batch_size": args.batch_size, "total_cases": len(cases),
        "interpretation": "Existing-checkpoint inference dependency ablation; not a rank-training ablation.",
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "total_cases": len(cases),
        "accuracy": {name: result["accuracy"] for name, result in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
