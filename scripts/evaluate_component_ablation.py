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


def _set_multipliers(model: GPT2WithARB, active_layers: set[str], lora: float) -> None:
    for layer, injector in model.injectors.items():
        injector.inject.set_eval_gate_multiplier(1.0 if layer in active_layers else 0.0)
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
    parser.add_argument("--setting", choices=[
        "full", "final_only", "layer_20_only", "injection_off", "lora_off", "both_off",
    ],
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

    layers = sorted(model.injectors.keys(), key=int)
    if len(layers) < 2:
        raise ValueError("Layer-position ablation requires at least two ARB injectors")
    settings = {
        "full": (set(layers), 1.0),
        "final_only": ({layers[-1]}, 1.0),
        "layer_20_only": ({"20"}, 1.0),
        "injection_off": (set(), 1.0),
        "lora_off": (set(layers), 0.0),
        "both_off": (set(), 0.0),
    }
    results = {}
    selected_settings = {args.setting: settings[args.setting]} if args.setting else settings
    for name, (active_layers, lora) in selected_settings.items():
        _set_multipliers(model, active_layers, lora)
        score = evaluator._evaluate_integer_cases(cases, include_examples=True)
        results[name] = {
            "active_injection_layers": sorted(active_layers, key=int),
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
