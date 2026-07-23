#!/usr/bin/env python3
"""Run reproducible ARB domain and clean-arithmetic evaluation slices.

The output keeps every prompt, generation, extracted answer, and correctness
flag so a rebuttal table can be checked without re-running the model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mathllm.evaluation.evaluator import ARBEvaluator
from mathllm.model.gpt2_arb import GPT2WithARB


RNS_PRIMES = (7, 11, 13, 17, 19, 23)
TWO_PRIME_FACTORS = {"7x11": 7 * 11, "11x13": 11 * 13}


def _integer_case(evaluator: ARBEvaluator, prompt: str, expected: int) -> dict[str, object]:
    return evaluator._build_integer_case(prompt, expected)


def build_division_slices(
    evaluator: ARBEvaluator,
    samples_per_slice: int,
    seed: int,
    labels: set[str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build exact divisions whose divisor is singular in selected RNS lanes."""
    rng = random.Random(seed)
    factors = {str(p): p for p in RNS_PRIMES} | TWO_PRIME_FACTORS
    slices: dict[str, list[dict[str, object]]] = {}
    for label, factor in factors.items():
        if labels is not None and label not in labels:
            continue
        cases = []
        for _ in range(samples_per_slice):
            divisor = factor * rng.randint(1, 999 // factor)
            quotient = rng.randint(1, 999)
            dividend = divisor * quotient
            cases.append(_integer_case(evaluator, f"{dividend}/{divisor}=", quotient))
        slices[label] = cases
    return slices


def build_clean_arithmetic_cells(
    evaluator: ARBEvaluator,
    samples_per_cell: int,
    seed: int,
    operations: set[str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build the ordinary 1--3 digit operation grid used for regression checks."""
    rng = random.Random(seed)
    cells: dict[str, list[dict[str, object]]] = {}
    for operation, symbol in (("add", "+"), ("sub", "-"), ("mul", "*")):
        if operations is not None and operation not in operations:
            continue
        for left_digits in range(1, 4):
            for right_digits in range(1, 4):
                low_a, high_a = 10 ** (left_digits - 1), 10**left_digits - 1
                low_b, high_b = 10 ** (right_digits - 1), 10**right_digits - 1
                cases = []
                for _ in range(samples_per_cell):
                    a = rng.randint(low_a, high_a)
                    b = rng.randint(low_b, high_b)
                    if operation == "sub" and a < b:
                        a, b = b, a
                    expected = a + b if operation == "add" else a - b if operation == "sub" else a * b
                    cases.append(_integer_case(evaluator, f"{a}{symbol}{b}=", expected))
                cells[f"{operation}_{left_digits}x{right_digits}"] = cases
    return cells


def score_cells(
    evaluator: ARBEvaluator,
    cells: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    results: dict[str, object] = {}
    for label, cases in cells.items():
        results[label] = evaluator._evaluate_integer_cases(cases, include_examples=True)
    return results


def _summary(results: dict[str, object]) -> dict[str, object]:
    correct = sum(int(result["correct"]) for result in results.values())
    total = sum(int(result["total"]) for result in results.values())
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="trained_model_360m")
    parser.add_argument("--output", default="review/artifacts/domain_slices.json")
    parser.add_argument("--division-samples", type=int, default=100)
    parser.add_argument("--arithmetic-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--division-slice", action="append", choices=[*map(str, RNS_PRIMES), *TWO_PRIME_FACTORS],
                        help="Evaluate only this division slice; repeat to select multiple.")
    parser.add_argument("--arithmetic-operation", action="append", choices=["add", "sub", "mul"],
                        help="Evaluate only this clean arithmetic operation; repeat to select multiple.")
    parser.add_argument("--skip-division", action="store_true")
    parser.add_argument("--skip-arithmetic", action="store_true")
    args = parser.parse_args()

    if args.division_samples < 1 or args.arithmetic_samples < 1:
        parser.error("sample counts must be positive")

    device = torch.device(args.device)
    model, tokenizer, config = GPT2WithARB.from_exported_model(args.model_dir, device=device)
    config.evaluation.batch_size = args.batch_size
    evaluator = ARBEvaluator(model, tokenizer, config.evaluation, device)

    division = {} if args.skip_division else score_cells(
        evaluator, build_division_slices(
            evaluator, args.division_samples, args.seed,
            set(args.division_slice) if args.division_slice else None,
        ),
    )
    arithmetic = {} if args.skip_arithmetic else score_cells(
        evaluator, build_clean_arithmetic_cells(
            evaluator, args.arithmetic_samples, args.seed + 1,
            set(args.arithmetic_operation) if args.arithmetic_operation else None,
        ),
    )
    report = {
        "date": str(date.today()),
        "model_dir": str(args.model_dir),
        "device": str(device),
        "seed": args.seed,
        "division_samples_per_slice": args.division_samples,
        "arithmetic_samples_per_cell": args.arithmetic_samples,
        "division": {"summary": _summary(division), "slices": division},
        "clean_arithmetic": {"summary": _summary(arithmetic), "cells": arithmetic},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "division": report["division"]["summary"],
        "clean_arithmetic": report["clean_arithmetic"]["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
