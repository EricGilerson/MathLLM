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

    # Count examples with auxiliary targets
    aux_count = sum(1 for ex in examples if ex.operand_a is not None)
    logger.info(f"  Examples with aux targets: {aux_count}/{len(examples)}")

    # Show digit distribution and operation breakdown if digit_weights are configured
    if config.data.digit_weights:
        from collections import Counter
        op_counts = Counter()
        digit_counts = Counter()
        for ex in examples:
            if ex.operand_a is not None:
                op_counts[ex.op_type] += 1
                max_op = max(len(str(abs(ex.operand_a))), len(str(abs(ex.operand_b))))
                digit_counts[max_op] += 1
        total = sum(digit_counts.values()) or 1
        logger.info(f"\nDigit distribution (weights={list(config.data.digit_weights)}):")
        for d in sorted(digit_counts):
            logger.info(f"  {d}-digit: {digit_counts[d]/total:.1%} ({digit_counts[d]})")
        logger.info("Operation breakdown:")
        for op in sorted(op_counts):
            logger.info(f"  {op}: {op_counts[op]}")

    # Print a few samples
    logger.info("\nSample examples:")
    for ex in examples[:10]:
        aux_tag = f" [a={ex.operand_a}, b={ex.operand_b}]" if ex.operand_a is not None else " [no aux]"
        logger.info(f"  {ex.text}{aux_tag}")


if __name__ == "__main__":
    main()
