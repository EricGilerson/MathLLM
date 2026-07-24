#!/usr/bin/env python3
"""Prepare and run a matched scratch-pretraining baseline/ARB seed sequentially.

Example:
  python scripts/run_toy_seed.py --config configs/toy_pretrain_gpu_seed2.yaml --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathllm.pretraining.experiment import load_toy_config, prepare_data, run_training


_SUMMARY_KEYS = (
    "parameter_count",
    "heldout_prose_nll",
    "heldout_prose_ppl",
    "heldout_direct_arithmetic_nll",
    "heldout_contextual_arithmetic_nll",
    "direct_arithmetic_accuracy",
    "contextual_seen_arithmetic_accuracy",
    "contextual_unseen_arithmetic_accuracy",
    "heldout_fact_nll",
    "heldout_fact_accuracy",
    "fact_eval_cases",
)


def _compact(metrics: dict) -> dict:
    return {key: metrics[key] for key in _SUMMARY_KEYS if key in metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Seed-specific toy-pretraining YAML")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse the configured existing mixture/tokenizer instead of preparing it again.",
    )
    args = parser.parse_args()

    config = load_toy_config(args.config)
    config.training.device = args.device

    if args.skip_prepare:
        print("Reusing existing prepared mixture.")
    else:
        print("Preparing deterministic seed mixture once for both variants.")
        metadata = prepare_data(config)["metadata"]
        print(json.dumps(metadata, indent=2))

    for variant in ("baseline", "arb"):
        print(f"\n===== {variant.upper()} =====")
        metrics = run_training(config, variant, prepare=False)
        print(json.dumps(_compact(metrics), indent=2))


if __name__ == "__main__":
    main()
