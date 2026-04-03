#!/usr/bin/env python3
"""Generate synthetic arithmetic training data."""

import argparse
import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from mathllm.config import load_config
from mathllm.data.generator import ArithmeticDataGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate arithmetic training data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config.data.output_dir = args.output_dir

    logger.info("Generating training data...")
    logger.info(f"  Positive examples: {config.data.num_positive}")
    logger.info(f"  Negative examples: {config.data.num_negative}")
    logger.info(f"  Edge cases: {config.data.num_edge_cases}")

    generator = ArithmeticDataGenerator(config.data)
    examples = generator.generate_dataset()

    output_path = generator.save_dataset(examples, config.data.output_dir)
    logger.info(f"Saved {len(examples)} examples to {output_path}")

    # Print a few samples
    logger.info("\nSample examples:")
    for ex in examples[:10]:
        logger.info(f"  {ex}")


if __name__ == "__main__":
    main()
