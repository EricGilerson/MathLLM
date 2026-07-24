#!/usr/bin/env python3
"""Evaluate an instruction-tuned causal LM with its declared chat template.

This is intentionally a separate protocol from the direct ``A op B =``
diagnostic. It reuses ARBEvaluator's deterministic arithmetic cells and answer
parser, while formatting each cell through the checkpoint tokenizer's chat
template before generation.
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
from mathllm.model.utils import get_device


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class RawCausalLMAdapter:
    """Expose a raw Hugging Face model through ARBEvaluator's small interface."""

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


class ChatTemplateEvaluator(ARBEvaluator):
    """Format each direct arithmetic cell as an Instruct user message."""

    instruction = "Compute this arithmetic expression. Return only the integer answer: "

    def _generate_texts(self, prompts, max_new_tokens=None):
        chat_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": f"{self.instruction}{prompt}"}]
            chat_prompts.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        return super()._generate_texts(chat_prompts, max_new_tokens=max_new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/360m_instruct_eval.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.batch_size is not None:
        config.evaluation.batch_size = args.batch_size
    device = get_device(config.training.device)
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.training.base_model)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    evaluator = ChatTemplateEvaluator(RawCausalLMAdapter(model), tokenizer, config.evaluation, device=device)
    results = evaluator.full_evaluation()
    report = {
        "model": config.training.base_model,
        "protocol": "chat_template",
        "message": ChatTemplateEvaluator.instruction + "<A op B =>",
        "chat_template": getattr(tokenizer, "chat_template", None),
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
