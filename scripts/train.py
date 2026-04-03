#!/usr/bin/env python3
"""Train ARB parameters on synthetic arithmetic data."""

import argparse
import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from mathllm.config import load_config
from mathllm.data.dataset import ArithmeticDataset
from mathllm.data.generator import ArithmeticDataGenerator
from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.model.utils import count_parameters, get_device
from mathllm.training.trainer import ARBTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train ARB-augmented GPT-2")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to pre-generated JSONL data. If not provided, generates on the fly.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.training.device)
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.training.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load or generate data
    if args.data_path:
        logger.info(f"Loading data from {args.data_path}")
        dataset = ArithmeticDataset(
            jsonl_path=args.data_path,
            tokenizer=tokenizer,
            max_length=config.training.max_seq_len,
        )
    else:
        logger.info("Generating training data on the fly...")
        generator = ArithmeticDataGenerator(config.data)
        examples = generator.generate_dataset()
        logger.info(f"Generated {len(examples)} examples")
        dataset = ArithmeticDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_length=config.training.max_seq_len,
        )

    # Split into train/eval
    train_ds, eval_ds = dataset.split(train_ratio=0.9)
    logger.info(f"Train: {len(train_ds)} examples, Eval: {len(eval_ds)} examples")

    train_loader = DataLoader(
        train_ds, batch_size=config.training.batch_size, shuffle=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=config.training.batch_size
    )

    # Build model
    logger.info(f"Loading base model: {config.training.base_model}")
    model = GPT2WithARB(config)
    params = count_parameters(model)
    logger.info(f"Parameters — trainable: {params['trainable']:,}, frozen: {params['frozen']:,}")

    # Create trainer
    trainer = ARBTrainer(
        model=model,
        config=config.training,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    history = trainer.train()
    logger.info("Training complete!")

    # Print final metrics
    if history["train_loss"]:
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history["eval_loss"]:
        logger.info(f"Best eval loss: {trainer.best_eval_loss:.4f}")


if __name__ == "__main__":
    main()
