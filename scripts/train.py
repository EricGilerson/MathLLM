#!/usr/bin/env python3
"""Train ARB parameters on synthetic arithmetic data."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from mathllm.config import load_config
from mathllm.data.dataset import ArithmeticDataset
from mathllm.data.generator import ArithmeticDataGenerator
from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.model.utils import count_parameters, get_device
from mathllm.training.trainer import ARBTrainer, resolve_resume_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train ARB-augmented GPT-2")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to pre-generated JSONL data. If not provided, generates on the fly.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable automatic resume from the latest checkpoint")
    parser.add_argument("--export-dir", type=str, default=None,
                        help="Directory to save the final exported model bundle")
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument("--epochs-to-run", type=int, default=None,
                              help="Number of additional epochs to run this invocation")
    budget_group.add_argument("--steps-to-run", type=int, default=None,
                              help="Number of additional optimizer steps to run this invocation")
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
        data_path = Path(args.data_path)
    else:
        # Auto-detect pre-generated data from scripts/generate_data.py
        default_data_path = Path(config.data.output_dir) / "train.jsonl"
        if default_data_path.exists():
            data_path = default_data_path
            logger.info(f"Found existing training data at {data_path}")
        else:
            data_path = None

    if data_path is not None:
        logger.info(f"Loading data from {data_path}")
        dataset = ArithmeticDataset(
            jsonl_path=data_path,
            tokenizer=tokenizer,
            max_length=config.training.max_seq_len,
            answer_only_loss=config.training.answer_only_loss,
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
            answer_only_loss=config.training.answer_only_loss,
        )

    # Split into train/eval
    train_ds, eval_ds = dataset.split(train_ratio=0.9)
    logger.info(f"Train: {len(train_ds)} examples, Eval: {len(eval_ds)} examples")

    use_pin_memory = device.type == "cuda"
    eval_bs = config.training.eval_batch_size or config.training.batch_size * 2

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=use_pin_memory,
        persistent_workers=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=eval_bs,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=use_pin_memory,
        persistent_workers=True,
    )

    # Build model
    logger.info(f"Loading base model: {config.training.base_model}")
    model = GPT2WithARB(config)
    params = count_parameters(model)
    logger.info(f"Parameters — trainable: {params['trainable']:,}, frozen: {params['frozen']:,}")

    # Compile forward pass for fused kernels (PyTorch 2.0+, CUDA only)
    import torch._dynamo
    torch._dynamo.config.verbose = False
    torch._dynamo.config.suppress_errors = True
    if device.type == "cuda" and hasattr(torch, "compile"):
        logger.info("Compiling model forward pass with torch.compile...")
        model.forward = torch.compile(model.forward, mode="reduce-overhead")

    # Create trainer
    trainer = ARBTrainer(
        model=model,
        config=config.training,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
    )

    resume_path = resolve_resume_checkpoint(
        checkpoint_dir=config.training.checkpoint_dir,
        explicit_resume=args.resume,
        auto_resume_latest=config.training.auto_resume_latest,
        disable_resume=args.no_resume,
    )
    if resume_path is not None:
        if args.resume:
            logger.info(f"Resuming from explicit checkpoint: {resume_path}")
        else:
            logger.info(f"Auto-resuming from latest checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)

    # Train
    logger.info("Starting training...")
    history = trainer.train(
        epochs_to_run=args.epochs_to_run,
        steps_to_run=args.steps_to_run,
    )
    logger.info("Training complete!")

    # Print final metrics
    if history["train_loss"]:
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history["eval_loss"]:
        logger.info(f"Best eval loss: {trainer.best_eval_loss:.4f}")

    export_dir = Path(args.export_dir or config.training.final_model_dir)
    config.training.final_model_dir = str(export_dir)
    logger.info(f"Exporting final model to {export_dir}")
    model.save_exported_model(export_dir, tokenizer)
    logger.info(f"Final model export saved to {export_dir}")


if __name__ == "__main__":
    main()
