#!/usr/bin/env python3
"""Train one fully-trainable toy baseline or ARB decoder from scratch."""

from __future__ import annotations

import argparse
import json

from mathllm.pretraining.experiment import load_toy_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/toy_pretrain_smoke.yaml")
    parser.add_argument("--variant", choices=["baseline", "arb"], required=True)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    config = load_toy_config(args.config)
    if args.device:
        config.training.device = args.device
    metrics = run_training(config, args.variant, prepare=args.prepare_data)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
