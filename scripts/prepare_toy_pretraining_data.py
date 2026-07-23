#!/usr/bin/env python3
"""Gather cached prose and generate a deterministic toy-pretraining mixture."""

from __future__ import annotations

import argparse
import json

from mathllm.pretraining.experiment import load_toy_config, prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/toy_pretrain_smoke.yaml")
    args = parser.parse_args()
    mixture = prepare_data(load_toy_config(args.config))
    print(json.dumps(mixture["metadata"], indent=2))


if __name__ == "__main__":
    main()
