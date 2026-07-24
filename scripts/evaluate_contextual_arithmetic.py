#!/usr/bin/env python3
"""Evaluate ARB or its matching raw base with a fixed instruction prefix.

This is a contextual compatibility condition, not a learned-invocation test:
the direct equation still appears at the end of every prompt and deterministically
triggers ARB when an ARB export is evaluated. It reuses the same seeded cells,
greedy decoding, and first-integer parser as ``scripts/evaluate.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mathllm.config import load_config
from mathllm.evaluation.evaluator import ARBEvaluator
from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.model.utils import get_device


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "Compute this arithmetic expression. Return only the integer answer: "


class RawCausalLMAdapter:
    """Adapt a raw Hugging Face causal LM to ARBEvaluator's interface."""

    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    def generate(self, input_ids, max_new_tokens=20, greedy=True):
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = self.model.config.pad_token_id or self.model.config.eos_token_id
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            pad_token_id=pad_token_id,
        )

    def __call__(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return {"loss": outputs.loss, "logits": outputs.logits}


class ContextualArithmeticEvaluator(ARBEvaluator):
    """Prepend one fixed instruction while preserving the direct equation."""

    def __init__(self, *args, prefix: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix

    def _generate_texts(self, prompts, max_new_tokens=None):
        return super()._generate_texts(
            [self.prefix + prompt for prompt in prompts],
            max_new_tokens=max_new_tokens,
        )


def _load_raw_base(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return RawCausalLMAdapter(model), tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/360m.yaml")
    load_group = parser.add_mutually_exclusive_group(required=True)
    load_group.add_argument("--model-dir", help="Exported ARB model directory")
    load_group.add_argument("--base-model-only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.batch_size is not None:
        config.evaluation.batch_size = args.batch_size
    device = get_device(config.training.device)
    logger.info("Using device: %s", device)

    if args.model_dir:
        model, tokenizer, exported_config = GPT2WithARB.from_exported_model(args.model_dir, device=device)
        model_identifier = exported_config.training.base_model
        condition = "exported_arb_contextual"
    else:
        model, tokenizer = _load_raw_base(config.training.base_model)
        model_identifier = config.training.base_model
        condition = "raw_base_contextual"

    evaluator = ContextualArithmeticEvaluator(
        model, tokenizer, config.evaluation, device=device, prefix=args.prefix,
    )
    results = evaluator.full_evaluation()
    report = {
        "model": model_identifier,
        "condition": condition,
        "prompt_protocol": "fixed_instruction_prefix_plus_direct_equation",
        "prefix": args.prefix,
        "equation_position": "prompt boundary",
        "decoding": "greedy",
        "evaluation": {
            "num_samples_per_config": config.evaluation.num_samples_per_config,
            "max_digits_range": config.evaluation.max_digits_range,
            "max_new_tokens": config.evaluation.max_new_tokens,
            "batch_size": config.evaluation.batch_size,
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    logger.info("Results saved to %s", output)
    logger.info(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
