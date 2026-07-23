#!/usr/bin/env python3
"""Measure end-to-end greedy-generation overhead of ARB versus its base LM."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

import torch

from mathllm.model.gpt2_arb import GPT2WithARB


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


def _measure(model, input_ids, max_new_tokens, is_arb, pad_token_id, warmup, repetitions):
    for _ in range(warmup):
        _generate(model, input_ids, max_new_tokens, is_arb, pad_token_id)
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = _generate(model, input_ids, max_new_tokens, is_arb, pad_token_id)
        elapsed = time.perf_counter() - start
        generated = int(output.size(1) - input_ids.size(1))
        samples.append({"latency_seconds": elapsed, "generated_tokens": generated, "tokens_per_second": generated / elapsed})
    return samples


def _summary(samples):
    latencies = [sample["latency_seconds"] for sample in samples]
    throughputs = [sample["tokens_per_second"] for sample in samples]
    return {
        "repetitions": len(samples), "latency_seconds": latencies,
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_median_seconds": statistics.median(latencies),
        "latency_min_seconds": min(latencies), "latency_max_seconds": max(latencies),
        "tokens_per_second_mean": statistics.mean(throughputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="trained_model_360m")
    parser.add_argument("--output", default="review/artifacts/runtime.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    arb, tokenizer, _ = GPT2WithARB.from_exported_model(args.model_dir, device=device)
    arb.eval()
    base = arb.base_model
    base.eval()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rows = []
    for length in args.prompt_lengths:
        ids = _prompt_ids(tokenizer, length, device)
        base_summary = _summary(_measure(base, ids, args.max_new_tokens, False, pad_token_id, args.warmup, args.repetitions))
        arb_summary = _summary(_measure(arb, ids, args.max_new_tokens, True, pad_token_id, args.warmup, args.repetitions))
        rows.append({
            "prompt_tokens": length, "base": base_summary, "arb": arb_summary,
            "latency_ratio_arb_over_base": arb_summary["latency_mean_seconds"] / base_summary["latency_mean_seconds"],
            "throughput_ratio_arb_over_base": arb_summary["tokens_per_second_mean"] / base_summary["tokens_per_second_mean"],
        })
    report = {
        "date": str(date.today()), "model_dir": args.model_dir, "device": str(device),
        "decoding": "greedy", "batch_size": 1, "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup, "repetitions": args.repetitions, "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
