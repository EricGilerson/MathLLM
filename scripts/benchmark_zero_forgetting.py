#!/usr/bin/env python3
"""Compare frozen base vs. ARB-augmented model on language retention benchmarks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mathllm.config import load_config
from mathllm.evaluation.zero_forgetting import run_zero_forgetting_benchmark
from mathllm.model.gpt2_arb import EXPORT_CONFIG_FILENAME, GPT2WithARB
from mathllm.model.utils import get_device
from mathllm.training.trainer import ARBTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_base_model(config, device: torch.device):
    """Load the frozen pretrained base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.training.base_model)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    return model, tokenizer


def _load_arb_model(args, config, device: torch.device):
    """Load the ARB-augmented model from export or checkpoint."""
    model_dir = args.model_dir
    if model_dir is None and args.checkpoint is None:
        configured_model_dir = Path(config.training.final_model_dir)
        if configured_model_dir.is_dir() and (configured_model_dir / EXPORT_CONFIG_FILENAME).exists():
            model_dir = str(configured_model_dir)
            logger.info(
                "Using exported model from config.training.final_model_dir: %s",
                model_dir,
            )
        else:
            raise FileNotFoundError(
                "No exported ARB model found in config.training.final_model_dir and no "
                "--model-dir/--checkpoint was provided."
            )

    if model_dir:
        logger.info("Loading exported ARB model from: %s", model_dir)
        model, tokenizer, _ = GPT2WithARB.from_exported_model(model_dir, device=device)
        return model, tokenizer, model_dir

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading ARB architecture from base model: %s", config.training.base_model)
    model = GPT2WithARB(config)
    model.build_token_digit_tables(tokenizer)

    if args.checkpoint:
        logger.info("Loading ARB checkpoint: %s", args.checkpoint)
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
    return model, tokenizer, args.checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the zero-forgetting benchmark on base vs. ARB models",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to an exported final model directory",
    )
    load_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to an ARB checkpoint to load",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save results as JSON")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for MC scoring")
    parser.add_argument("--wikitext-limit", type=int, default=256, help="WikiText-103 documents")
    parser.add_argument("--piqa-limit", type=int, default=512, help="PIQA validation examples")
    parser.add_argument("--hellaswag-limit", type=int, default=512, help="HellaSwag validation examples")
    parser.add_argument(
        "--perplexity-max-length",
        type=int,
        default=512,
        help="Maximum context window for WikiText perplexity",
    )
    parser.add_argument(
        "--perplexity-stride",
        type=int,
        default=512,
        help="Stride for WikiText perplexity windows",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.training.device)
    logger.info("Using device: %s", device)

    base_model, base_tokenizer = _load_base_model(config, device)
    arb_model, arb_tokenizer, arb_source = _load_arb_model(args, config, device)

    results = run_zero_forgetting_benchmark(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        arb_model=arb_model,
        arb_tokenizer=arb_tokenizer,
        device=device,
        batch_size=args.batch_size,
        perplexity_max_length=args.perplexity_max_length,
        perplexity_stride=args.perplexity_stride,
        wikitext_limit=args.wikitext_limit,
        piqa_limit=args.piqa_limit,
        hellaswag_limit=args.hellaswag_limit,
    )

    results["base_model"] = config.training.base_model
    results["arb_source"] = arb_source

    logger.info("\n=== Zero-Forgetting Benchmark ===")
    logger.info("\n%s", results["markdown_table"])

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
