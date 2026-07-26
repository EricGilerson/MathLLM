#!/usr/bin/env python3
"""Evaluate a saved toy baseline or ARB checkpoint on held-out fact queries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathllm.pretraining.arithmetic_bpe_tokenizer import ArithmeticBPETokenizer
from mathllm.pretraining.experiment import _fact_metrics, _generate, build_model, load_toy_config, resolve_device
from mathllm.pretraining.fact_benchmark import (
    atomic_fact_eval_cases,
    compositional_fact_eval_cases,
    fact_seen_template_cases,
    make_atomic_facts,
    make_compositional_facts,
    make_facts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--variant", choices=("baseline", "arb", "both"), default="both",
        help="Checkpoint variant to evaluate; 'both' evaluates the paired run sequentially.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--show", type=int, default=0, help="Include this many generated fact-query examples.")
    parser.add_argument("--split", choices=("heldout", "seen"), default="heldout")
    args = parser.parse_args()
    config = load_toy_config(args.config)
    device = resolve_device(args.device)
    mixture = torch.load(config.data.mixture_file, map_location="cpu", weights_only=False)
    tokenizer = ArithmeticBPETokenizer.from_file(config.data.tokenizer_file)
    atomic = config.data.fact_format in {"atomic", "compositional"}
    if args.split == "seen" and not atomic:
        facts = make_facts(config.data.fact_count, config.training.seed + 5)
        cases = fact_seen_template_cases(256, config.training.seed + 9, facts)
    elif config.data.fact_format == "compositional":
        cases = compositional_fact_eval_cases(make_compositional_facts(
            config.data.fact_count, config.training.seed + 5,
            config.data.fact_key_vocab_size, config.data.fact_value_vocab_size,
        ))
    elif atomic:
        # The atomic probe has one canonical query format; its evaluation
        # split already tests stored bindings with no answer in the prompt.
        cases = atomic_fact_eval_cases(make_atomic_facts(config.data.fact_count, config.training.seed + 5))
    else:
        cases = mixture.get("fact_eval_cases", [])
    variants = ("baseline", "arb") if args.variant == "both" else (args.variant,)
    results = {}
    for variant in variants:
        checkpoint = args.checkpoint or Path(config.training.output_dir) / variant / "checkpoint.pt"
        if args.checkpoint and args.variant == "both":
            parser.error("--checkpoint may only be used with one explicit variant")
        model = build_model(config, variant, tokenizer).to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
        result = _fact_metrics(model, tokenizer, cases, device, atomic=atomic)
        if args.show:
            examples = []
            if atomic:
                prompts, expected = zip(*cases[:args.show])
                input_ids = torch.tensor([tokenizer.encode(prompt) for prompt in prompts], device=device)
                with torch.inference_mode():
                    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                    prediction = logits[:, -1, :].argmax(dim=-1).tolist()
                examples = [
                    {"prompt": prompt, "expected": target, "predicted": tokenizer.decode([token_id])}
                    for prompt, target, token_id in zip(prompts, expected, prediction)
                ]
            else:
                for prompt, expected in cases[:args.show]:
                    full = _generate(model, tokenizer, prompt, len(expected) + 8, device)
                    completion = full[len(prompt):]
                    examples.append({"prompt": prompt, "expected": expected, "completion": completion})
            result["examples"] = examples
        results[variant] = result
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps(results if args.variant == "both" else results[args.variant], indent=2))


if __name__ == "__main__":
    main()
