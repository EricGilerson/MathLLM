#!/usr/bin/env python3
"""Export an inference-ready model bundle from a training checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mathllm.model.utils import get_device
from scripts.infer_checkpoint import load_checkpointed_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def export_checkpoint_model(
    config_path: str,
    checkpoint_path: str | None,
    output_dir: str | None,
    device: torch.device,
) -> tuple[Path, Path]:
    """Build a full model from checkpoint weights and export a standalone bundle."""
    model, tokenizer, config, resolved_checkpoint = load_checkpointed_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    export_dir = Path(output_dir or config.training.final_model_dir)
    config.training.final_model_dir = str(export_dir)
    exported_path = model.save_exported_model(export_dir, tokenizer)
    return exported_path, resolved_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an inference-ready model bundle from a training checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML used for training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to an ARB checkpoint. Defaults to the latest checkpoint.",
    )
    parser.add_argument(
        "--export-dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Destination for the exported model bundle. Defaults to training.final_model_dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device override: auto, cpu, cuda, or mps",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    logger.info("Using device: %s", device)

    export_dir, checkpoint_path = export_checkpoint_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=device,
    )
    logger.info("Loaded checkpoint from: %s", checkpoint_path)
    logger.info("Exported model bundle to: %s", export_dir)


if __name__ == "__main__":
    main()
