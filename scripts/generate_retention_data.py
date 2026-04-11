#!/usr/bin/env python3
"""Generate a language retention dataset from built-in negative templates.

Creates a plain text file with diverse non-arithmetic text to prevent
the LoRA head from overriding the base model's language capability.
Uses the existing negative example templates (170+ across 34 categories).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathllm.data.negative_examples import NegativeExampleSampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate language retention data"
    )
    parser.add_argument(
        "--output", type=str, default="data/retention.txt",
        help="Output file path (one example per line)",
    )
    parser.add_argument(
        "--count", type=int, default=100000,
        help="Number of examples to generate",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sampler = NegativeExampleSampler(seed=args.seed)
    examples = sampler.sample(args.count)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(ex.strip() + "\n")

    print(f"Wrote {len(examples)} retention examples to {output_path}")


if __name__ == "__main__":
    main()
