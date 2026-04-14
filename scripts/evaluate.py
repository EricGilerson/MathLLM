#!/usr/bin/env python3
"""Evaluate an ARB-augmented GPT-2 model."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

from mathllm.config import load_config
from mathllm.evaluation.evaluator import ARBEvaluator
from mathllm.model.gpt2_arb import EXPORT_CONFIG_FILENAME, GPT2WithARB
from mathllm.model.utils import get_device
from mathllm.training.trainer import ARBTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARB-augmented GPT-2")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument("--model-dir", type=str, default=None,
                            help="Path to an exported final model directory")
    load_group.add_argument("--checkpoint", type=str, default=None,
                            help="Path to ARB checkpoint to load")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results as JSON")
    parser.add_argument("--eval-texts", type=str, default=None,
                        help="Path to text file with eval texts for perplexity (one per line)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override evaluation batch size for generation and perplexity")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.batch_size is not None:
        config.evaluation.batch_size = args.batch_size
    device = get_device(config.training.device)
    logger.info(f"Using device: {device}")

    model_dir = args.model_dir
    if model_dir is None and args.checkpoint is None:
        configured_model_dir = Path(config.training.final_model_dir)
        if configured_model_dir.is_dir() and (configured_model_dir / EXPORT_CONFIG_FILENAME).exists():
            model_dir = str(configured_model_dir)
            logger.info(
                "Using exported model from config.training.final_model_dir: %s",
                model_dir,
            )

    if model_dir:
        logger.info(f"Loading exported model from: {model_dir}")
        model, tokenizer, _ = GPT2WithARB.from_exported_model(
            model_dir,
            device=device,
        )
    else:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model: {config.training.base_model}")
        model = GPT2WithARB(config)
        model.build_token_digit_tables(tokenizer)

        # Load checkpoint if provided
        if args.checkpoint:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            arb_state = ARBTrainer._migrate_legacy_arb_state(ckpt["arb_state"])
            model.compute_core.load_state_dict(arb_state["compute_core"], strict=False)
            for key, state in arb_state["injectors"].items():
                model.injectors[key].load_state_dict(state, strict=False)
            if "lora_state" in ckpt and model.lora_head is not None:
                model.lora_head.load_state_dict(ckpt["lora_state"])
            if "lora_layers_state" in ckpt and model.lora_layers is not None:
                model.lora_layers.load_state_dict(ckpt["lora_layers_state"])

        model.to(device)

    # Load eval texts for perplexity if provided
    eval_texts = None
    if args.eval_texts:
        with open(args.eval_texts) as f:
            eval_texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(eval_texts)} eval texts for perplexity")

    # Run evaluation
    evaluator = ARBEvaluator(model, tokenizer, config.evaluation, device=device)
    results = evaluator.full_evaluation(eval_texts=eval_texts)

    # Print results
    logger.info("\n=== Results ===")
    logger.info(json.dumps(results, indent=2, default=str))

    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
