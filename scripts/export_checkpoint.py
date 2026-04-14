#!/usr/bin/env python3
"""Export an inference-ready model bundle from a training checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mathllm.config import load_config
from mathllm.model.utils import get_device
from scripts.infer_checkpoint import load_checkpointed_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def resolve_export_checkpoint_path(
    config_path: str,
    checkpoint_path: str | None,
    use_best: bool = False,
) -> str | None:
    """Resolve which checkpoint should be exported."""
    if checkpoint_path is not None:
        return checkpoint_path
    if not use_best:
        return None

    config = load_config(config_path)
    best_checkpoint = Path(config.training.checkpoint_dir) / "arb_best.pt"
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_checkpoint}"
        )
    return str(best_checkpoint)


def export_checkpoint_model(
    config_path: str,
    checkpoint_path: str | None,
    output_dir: str | None,
    device: torch.device,
    use_best: bool = False,
) -> tuple[Path, Path]:
    """Build a full model from checkpoint weights and export a standalone bundle."""
    resolved_checkpoint_arg = resolve_export_checkpoint_path(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        use_best=use_best,
    )
    model, tokenizer, config, resolved_checkpoint = load_checkpointed_model(
        config_path=config_path,
        checkpoint_path=resolved_checkpoint_arg,
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
    parser.add_argument(
        "--best",
        action="store_true",
        help="Export checkpoints/arb_best.pt from the config checkpoint dir instead of the latest checkpoint.",
    )
    args = parser.parse_args()

    if args.best and args.checkpoint is not None:
        parser.error("--best cannot be used together with --checkpoint")

    device = get_device(args.device)
    logger.info("Using device: %s", device)

    export_dir, checkpoint_path = export_checkpoint_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=device,
        use_best=args.best,
    )
    logger.info("Loaded checkpoint from: %s", checkpoint_path)
    logger.info("Exported model bundle to: %s", export_dir)


if __name__ == "__main__":
    main()
