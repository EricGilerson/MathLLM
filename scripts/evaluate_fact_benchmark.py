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
from mathllm.pretraining.experiment import _fact_metrics, build_model, load_toy_config, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("baseline", "arb"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    config = load_toy_config(args.config)
    device = resolve_device(args.device)
    mixture = torch.load(config.data.mixture_file, map_location="cpu", weights_only=False)
    tokenizer = ArithmeticBPETokenizer.from_file(config.data.tokenizer_file)
    model = build_model(config, args.variant, tokenizer).to(device).eval()
    checkpoint = args.checkpoint or Path(config.training.output_dir) / args.variant / "checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
    print(json.dumps(_fact_metrics(model, tokenizer, mixture.get("fact_eval_cases", []), device), indent=2))


if __name__ == "__main__":
    main()
