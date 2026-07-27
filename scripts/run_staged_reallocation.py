#!/usr/bin/env python3
"""Run the staged arithmetic-retention versus fact-capacity experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathllm.pretraining.capacity_displacement import (
    load_displacement_config,
    run_experiment,
)


DEFAULT_CONDITIONS = ("fact_only", "baseline_mixed", "arb_mixed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("fact_only", "baseline_mixed", "arb_mixed", "zero_arb_mixed"),
        default=list(DEFAULT_CONDITIONS),
    )
    args = parser.parse_args()
    config = load_displacement_config(args.config)
    config.training.device = args.device
    summary = run_experiment(
        config,
        prepare=args.prepare,
        conditions=args.conditions,
    )
    compact = {}
    for condition, metrics in summary["results"].items():
        compact[condition] = {
            key: value for key, value in metrics.items()
            if key not in {
                "training_history",
                "evaluation_history",
                "arithmetic_pretrain_history",
            }
        }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
