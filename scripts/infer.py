#!/usr/bin/env python3
"""Run inference with an exported ARB model bundle."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.model.utils import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROMPT_PATTERN = re.compile(
    r"^\s*(-?\d+)\s*(\+|-|\*|\*\*|/)\s*(-?\d+)\s*=\s*$"
)


def compute_expected(prompt: str) -> int | None:
    """Return the exact expected integer result for simple arithmetic prompts."""
    match = _PROMPT_PATTERN.match(prompt)
    if not match:
        return None

    a = int(match.group(1))
    op = match.group(2)
    b = int(match.group(3))

    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "**":
        try:
            result = a**b
        except (OverflowError, ValueError):
            return None
        return result if abs(result) <= 10**12 else None
    if op == "/":
        if b == 0 or a % b != 0:
            return None
        return a // b
    return None


def generate_text(
    model: GPT2WithARB,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, str]:
    """Generate completion text for a prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    completion = full_text[len(prompt):]
    return full_text, completion


def print_result(prompt: str, completion: str, full_text: str) -> None:
    """Print a compact result summary."""
    expected = compute_expected(prompt)

    print(f"prompt:     {prompt}")
    print(f"completion: {completion.strip() or '<empty>'}")
    print(f"full_text:  {full_text}")
    if expected is not None:
        print(f"expected:   {expected}")
    print()


def interactive_loop(
    model: GPT2WithARB,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> None:
    """Run a simple prompt REPL."""
    print("Enter a prompt like '347 * 291 ='. Press Ctrl-D or submit an empty line to exit.\n")

    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            break

        if not prompt:
            break

        full_text, completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        print_result(prompt, completion, full_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with an exported ARB model bundle"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="trained_model/",
        help="Path to an exported model bundle",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run, e.g. '347 * 291 ='",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device override: auto, cpu, cuda, or mps",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Exported model directory not found: {model_dir}. "
            "Run scripts/train.py first or pass --model-dir."
        )

    device = get_device(args.device)
    logger.info("Using device: %s", device)
    logger.info("Loading exported model from: %s", model_dir)
    model, tokenizer, _ = GPT2WithARB.from_exported_model(model_dir, device=device)
    model.eval()

    if args.prompt is not None:
        full_text, completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        print_result(args.prompt, completion, full_text)
        return

    interactive_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
