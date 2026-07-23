#!/usr/bin/env python3
"""Benchmark base/ARB generation and an in-process CPU arithmetic calculator.

The calculator boundary is deliberately narrow and explicit: a Python string
containing ``A op B =`` is parsed, computed with Python integers on the host
CPU, and returned as a decimal string.  It includes no RPC, process launch,
tokenization, or model inference.  The corresponding ARB comparison measures
the full prompt-string -> tokenization -> GPU generation -> decoded-answer
path, with accelerator synchronization around each timing sample. A separate
synthetic tool-relay path measures GPU generation -> CPU handoff/calculation
-> resumed GPU generation. The base model is not tool-call trained, so tool
arguments are deterministically extracted from the direct input equation; the
relay is a latency measurement, not an agent-capability evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from datetime import date
from pathlib import Path

import torch

from mathllm.model.gpt2_arb import GPT2WithARB


_EQUATION = re.compile(r"^\s*(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*$")


def _synchronize(device: torch.device) -> None:
    """Ensure accelerator work is included in a wall-clock timing sample."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def in_process_cpu_calculator(prompt: str) -> str:
    """Parse one supported direct equation and return its integer answer.

    This intentionally uses Python scalar arithmetic, so it always executes on
    the CPU even when the accompanying model benchmark uses CUDA.
    """
    match = _EQUATION.fullmatch(prompt)
    if match is None:
        raise ValueError(f"Expected a direct equation 'A op B =', got {prompt!r}")
    a, operator, b = int(match.group(1)), match.group(2), int(match.group(3))
    if operator == "+":
        value = a + b
    elif operator == "-":
        value = a - b
    elif operator == "*":
        value = a * b
    else:
        if b == 0 or a % b:
            raise ValueError("Calculator benchmark only permits exact nonzero division")
        value = a // b
    return str(value)


def arithmetic_prompts(count: int, seed: int) -> list[tuple[str, str]]:
    """Create deterministic, supported direct equations for latency samples."""
    rng = random.Random(seed)
    prompts: list[tuple[str, str]] = []
    operations = ("+", "-", "*", "/")
    for index in range(count):
        operator = operations[index % len(operations)]
        if operator == "/":
            b, expected = rng.randint(1, 999), rng.randint(0, 999)
            a = b * expected
        else:
            a, b = rng.randint(0, 999), rng.randint(0, 999)
            if operator == "-" and a < b:
                a, b = b, a
            expected = int(in_process_cpu_calculator(f"{a} {operator} {b} ="))
        prompt = f"{a} {operator} {b} ="
        prompts.append((prompt, str(expected)))
    return prompts


def _prompt_ids(tokenizer, length: int, device: torch.device) -> torch.Tensor:
    seed = tokenizer.encode(" The quick brown fox observes a small experiment.", add_special_tokens=False)
    ids = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _generate(model, input_ids: torch.Tensor, max_new_tokens: int, is_arb: bool, pad_token_id: int) -> torch.Tensor:
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        if is_arb:
            return model.generate(input_ids, max_new_tokens=max_new_tokens, greedy=True)
        return model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_token_id,
        )


def _measure_generation(model, input_ids, max_new_tokens, is_arb, pad_token_id, warmup, repetitions, device):
    for _ in range(warmup):
        _generate(model, input_ids, max_new_tokens, is_arb, pad_token_id)
    _synchronize(device)
    samples = []
    for _ in range(repetitions):
        _synchronize(device)
        start = time.perf_counter()
        output = _generate(model, input_ids, max_new_tokens, is_arb, pad_token_id)
        _synchronize(device)
        elapsed = time.perf_counter() - start
        generated = int(output.size(1) - input_ids.size(1))
        samples.append({"latency_seconds": elapsed, "generated_tokens": generated, "tokens_per_second": generated / elapsed})
    return samples


def _measure_detection_only(detector, input_ids, eq_token_id, warmup, repetitions, device):
    """Time the token-level valid-equation detector alone (not tokenization)."""
    for _ in range(warmup):
        detector(input_ids, eq_token_id)
    _synchronize(device)
    samples = []
    for _ in range(repetitions):
        _synchronize(device)
        start = time.perf_counter()
        detector(input_ids, eq_token_id)
        _synchronize(device)
        samples.append({"latency_seconds": time.perf_counter() - start})
    return samples


def _measure_cpu_calculator(prompts: list[tuple[str, str]], warmup: int, repetitions: int) -> tuple[list[dict], int]:
    for prompt, _ in prompts[:warmup]:
        in_process_cpu_calculator(prompt)
    samples, correct = [], 0
    for _ in range(repetitions):
        for prompt, expected in prompts:
            start = time.perf_counter()
            answer = in_process_cpu_calculator(prompt)
            elapsed = time.perf_counter() - start
            correct += int(answer == expected)
            samples.append({"latency_seconds": elapsed, "operations_per_second": 1.0 / elapsed})
    return samples, correct


def _measure_arb_prompt_to_answer(arb, tokenizer, prompts, max_new_tokens, warmup, repetitions, device):
    """Measure text prompt to decoded answer, including CPU tokenization/H2D copy."""
    for prompt, _ in prompts[:warmup]:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        _generate(arb, ids, max_new_tokens, True, tokenizer.pad_token_id or tokenizer.eos_token_id)
    _synchronize(device)
    samples, correct = [], 0
    for _ in range(repetitions):
        for prompt, expected in prompts:
            _synchronize(device)
            start = time.perf_counter()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            output = _generate(arb, input_ids, max_new_tokens, True, tokenizer.pad_token_id or tokenizer.eos_token_id)
            continuation = tokenizer.decode(output[0, input_ids.size(1):], skip_special_tokens=True).strip()
            _synchronize(device)
            elapsed = time.perf_counter() - start
            correct += int(continuation.startswith(expected))
            samples.append({"latency_seconds": elapsed, "generated_tokens": int(output.size(1) - input_ids.size(1))})
    return samples, correct


def _measure_in_process_tool_relay(
    base,
    tokenizer,
    prompts: list[tuple[str, str]],
    pre_new_tokens: int,
    post_new_tokens: int,
    warmup: int,
    repetitions: int,
    device: torch.device,
):
    """Measure model generation -> CPU calculator -> resumed model generation.

    The generated pre-tool token is copied from GPU to CPU to model the
    handoff. The known direct input equation supplies the calculator argument,
    because this base model was not trained to emit a formal tool-call syntax.
    """
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def run_once(prompt: str, expected: str, record: bool):
        _synchronize(device)
        total_start = time.perf_counter()

        pre_start = time.perf_counter()
        prefix = f"Question: {prompt}\nCalculator call: "
        initial_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
        pre_output = _generate(base, initial_ids, pre_new_tokens, False, pad_token_id)
        _synchronize(device)
        pre_elapsed = time.perf_counter() - pre_start

        cpu_start = time.perf_counter()
        # This transfer and decode represent the generated call leaving the
        # accelerator. The direct prompt provides the deterministic tool args.
        generated_call = pre_output[0, initial_ids.size(1):].detach().cpu()
        _ = tokenizer.decode(generated_call, skip_special_tokens=True)
        calculator_result = in_process_cpu_calculator(prompt)
        result_ids = tokenizer.encode(
            f"\nCalculator result: {calculator_result}\nAnswer: ", return_tensors="pt",
        )
        cpu_elapsed = time.perf_counter() - cpu_start

        post_start = time.perf_counter()
        resumed_ids = torch.cat([pre_output, result_ids.to(device)], dim=1)
        post_output = _generate(base, resumed_ids, post_new_tokens, False, pad_token_id)
        _synchronize(device)
        post_elapsed = time.perf_counter() - post_start
        total_elapsed = time.perf_counter() - total_start

        if not record:
            return None
        continuation = tokenizer.decode(post_output[0, resumed_ids.size(1):], skip_special_tokens=True).strip()
        return {
            "latency_seconds": total_elapsed,
            "pre_tool_generation_seconds": pre_elapsed,
            "cpu_handoff_and_calculation_seconds": cpu_elapsed,
            "post_tool_generation_seconds": post_elapsed,
            "generated_tokens": int(pre_output.size(1) - initial_ids.size(1) + post_output.size(1) - resumed_ids.size(1)),
            "calculator_answer_correct": calculator_result == expected,
            "final_answer_prefix_match": continuation.startswith(expected),
        }

    for prompt, expected in prompts[:warmup]:
        run_once(prompt, expected, record=False)
    samples = []
    for _ in range(repetitions):
        for prompt, expected in prompts:
            samples.append(run_once(prompt, expected, record=True))
    return samples


def _summary(samples: list[dict], include_latency_samples: bool = True) -> dict:
    latencies = [sample["latency_seconds"] for sample in samples]
    summary = {
        "samples": len(samples),
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_median_seconds": statistics.median(latencies),
        "latency_min_seconds": min(latencies), "latency_max_seconds": max(latencies),
        "operations_per_second_mean": statistics.mean(1.0 / latency for latency in latencies),
    }
    if include_latency_samples:
        summary["latency_seconds"] = latencies
    if all("generated_tokens" in sample for sample in samples):
        generated = [sample["generated_tokens"] for sample in samples]
        summary["generated_tokens_mean"] = statistics.mean(generated)
        summary["tokens_per_second_mean"] = statistics.mean(tokens / latency for tokens, latency in zip(generated, latencies))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="trained_model_360m")
    parser.add_argument("--output", default="review/artifacts/runtime.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--calculator-cases", type=int, default=100)
    parser.add_argument("--detector-repetitions", type=int, default=1000)
    parser.add_argument("--tool-pre-new-tokens", type=int, default=1)
    parser.add_argument(
        "--tool-post-new-tokens", type=int, default=None,
        help="Post-tool base-model tokens to generate (defaults to --max-new-tokens)",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    if args.tool_post_new_tokens is None:
        args.tool_post_new_tokens = args.max_new_tokens
    if min(args.calculator_cases, args.repetitions, args.detector_repetitions, args.tool_pre_new_tokens, args.tool_post_new_tokens) <= 0:
        parser.error("case counts, repetitions, and tool-generation token counts must be positive")

    device = torch.device(args.device)
    arb, tokenizer, _ = GPT2WithARB.from_exported_model(args.model_dir, device=device)
    arb.eval()
    base = arb.base_model
    base.eval()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model_rows = []
    for length in args.prompt_lengths:
        ids = _prompt_ids(tokenizer, length, device)
        detection_summary = _summary(_measure_detection_only(
            arb.compute_core.extract.find_valid_equations, ids, arb._eq_token_id,
            args.warmup, args.detector_repetitions, device,
        ), include_latency_samples=False)
        base_summary = _summary(_measure_generation(base, ids, args.max_new_tokens, False, pad_token_id, args.warmup, args.repetitions, device))
        arb_summary = _summary(_measure_generation(arb, ids, args.max_new_tokens, True, pad_token_id, args.warmup, args.repetitions, device))
        model_rows.append({
            "prompt_tokens": length, "detector_only": detection_summary,
            "base": base_summary, "arb": arb_summary,
            "latency_ratio_arb_over_base": arb_summary["latency_mean_seconds"] / base_summary["latency_mean_seconds"],
            "throughput_ratio_arb_over_base": arb_summary["tokens_per_second_mean"] / base_summary["tokens_per_second_mean"],
        })

    prompts = arithmetic_prompts(args.calculator_cases, args.seed)
    calculator_samples, calculator_correct = _measure_cpu_calculator(prompts, args.warmup, args.repetitions)
    arb_samples, arb_correct = _measure_arb_prompt_to_answer(
        arb, tokenizer, prompts, args.max_new_tokens, args.warmup, args.repetitions, device,
    )
    tool_relay_samples = _measure_in_process_tool_relay(
        base, tokenizer, prompts, args.tool_pre_new_tokens, args.tool_post_new_tokens,
        args.warmup, args.repetitions, device,
    )
    calculator_summary, arb_arithmetic_summary = _summary(calculator_samples), _summary(arb_samples)
    tool_relay_summary = _summary(tool_relay_samples)
    tool_relay_summary.update({
        "pre_tool_generation_mean_seconds": statistics.mean(sample["pre_tool_generation_seconds"] for sample in tool_relay_samples),
        "cpu_handoff_and_calculation_mean_seconds": statistics.mean(sample["cpu_handoff_and_calculation_seconds"] for sample in tool_relay_samples),
        "post_tool_generation_mean_seconds": statistics.mean(sample["post_tool_generation_seconds"] for sample in tool_relay_samples),
    })
    total_cases = args.calculator_cases * args.repetitions
    report = {
        "date": str(date.today()), "model_dir": args.model_dir, "device": str(device),
        "base_vs_arb_boundary": "pre-tokenized, device-resident prompt IDs -> greedy generated IDs; ARB no-trigger rows include valid-equation detection and fast-path routing",
        "calculator_boundary": "CPU Python string parse -> integer arithmetic -> decimal string; no RPC, process launch, tokenizer, or model",
        "arb_calculator_comparison_boundary": "prompt string -> tokenizer + host-to-device copy + synchronized ARB greedy generation + decoded continuation",
        "decoding": "greedy", "batch_size": 1, "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup, "repetitions": args.repetitions,
        "detector_repetitions": args.detector_repetitions,
        "detector_boundary": "device-resident token IDs -> token-level valid-equation decision; excludes tokenizer and full ARB compute",
        "model_generation": model_rows,
        "in_process_cpu_calculator": {
            "cases_per_repetition": args.calculator_cases, "total_cases": total_cases,
            "exact_answers": calculator_correct, "summary": calculator_summary,
        },
        "arb_prompt_to_answer": {
            "cases_per_repetition": args.calculator_cases, "total_cases": total_cases,
            "answer_prefix_matches": arb_correct, "summary": arb_arithmetic_summary,
            "latency_ratio_arb_over_cpu_calculator": (
                arb_arithmetic_summary["latency_mean_seconds"] / calculator_summary["latency_mean_seconds"]
            ),
        },
        "synthetic_in_process_tool_relay": {
            "boundary": "GPU base generation -> generated-token GPU-to-CPU handoff + CPU calculator -> result CPU-to-GPU transfer + resumed GPU base generation",
            "tool_argument_source": "the already-present direct input equation; this model is not trained to generate callable tool syntax",
            "pre_tool_new_tokens": args.tool_pre_new_tokens,
            "post_tool_new_tokens": args.tool_post_new_tokens,
            "cases_per_repetition": args.calculator_cases,
            "total_cases": total_cases,
            "calculator_answers_correct": sum(sample["calculator_answer_correct"] for sample in tool_relay_samples),
            "final_answer_prefix_matches": sum(sample["final_answer_prefix_match"] for sample in tool_relay_samples),
            "summary": tool_relay_summary,
            "comparison_note": (
                "Compare total latency only after accounting for generated-token budgets: "
                "ARB uses max_new_tokens, while this relay uses pre_tool_new_tokens + post_tool_new_tokens."
            ),
            "latency_ratio_arb_over_tool_relay_with_configured_token_budgets": (
                arb_arithmetic_summary["latency_mean_seconds"] / tool_relay_summary["latency_mean_seconds"]
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
