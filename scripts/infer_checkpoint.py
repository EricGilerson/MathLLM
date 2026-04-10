#!/usr/bin/env python3
"""Run prompt debugging directly against an ARB checkpoint."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

from mathllm.config import Config, load_config
from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.model.utils import get_device
from mathllm.training.trainer import find_latest_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROMPT_PATTERN = re.compile(
    r"^\s*(-?\d+)\s*(\+|-|\*|\*\*|/)\s*(-?\d+)\s*=\s*$"
)


@dataclass(frozen=True)
class LayerExtraction:
    """Decoded extraction details for one ARB layer at one token position."""

    layer_id: int
    token_index: int
    digits_a: list[int]
    digits_b: list[int]
    value_a: int
    value_b: int


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


def digits_to_int(digits: list[int]) -> int:
    """Reconstruct an integer magnitude from LSB-first digits."""
    return sum(digit * (10 ** idx) for idx, digit in enumerate(digits))


def collect_layer_extractions(
    arb_extractions: dict[int, tuple[torch.Tensor, torch.Tensor]],
    eq_index: int,
) -> list[LayerExtraction]:
    """Decode deterministic digit extractions from each ARB layer.

    arb_extractions contain (d_a, d_b) digit vectors [B, S, K].
    All positions have the same values (broadcast), so eq_index is cosmetic.
    """
    layers: list[LayerExtraction] = []
    for layer_id, (d_a, d_b) in sorted(arb_extractions.items()):
        # All positions have the same digits; pick eq_index for display
        digits_a = d_a[0, eq_index].long().tolist()
        digits_b = d_b[0, eq_index].long().tolist()

        layers.append(
            LayerExtraction(
                layer_id=layer_id,
                token_index=eq_index,
                digits_a=digits_a,
                digits_b=digits_b,
                value_a=digits_to_int(digits_a),
                value_b=digits_to_int(digits_b),
            )
        )
    return layers


def format_extractions(extractions: list[LayerExtraction], op_index: int | None = None, eq_index: int | None = None) -> str:
    """Render decoded extraction details for printing."""
    if not extractions:
        return "extractions: <none>"

    lines = ["extractions:"]
    for extraction in extractions:
        a_pos = f"op@{op_index}" if op_index is not None else f"@{extraction.token_index}"
        b_pos = f"eq@{eq_index}" if eq_index is not None else f"@{extraction.token_index}"
        lines.append(
            f"  layer {extraction.layer_id}: "
            f"a({a_pos})={extraction.digits_a} val={extraction.value_a}  "
            f"b({b_pos})={extraction.digits_b} val={extraction.value_b}"
        )
    return "\n".join(lines)


def resolve_checkpoint_path(
    config: Config,
    explicit_checkpoint: str | None,
) -> Path:
    """Resolve which checkpoint to inspect."""
    if explicit_checkpoint is not None:
        path = Path(explicit_checkpoint)
    else:
        latest = find_latest_checkpoint(config.training.checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint found in {config.training.checkpoint_dir}"
            )
        path = latest

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpointed_model(
    config_path: str,
    checkpoint_path: str | None,
    device: torch.device,
) -> tuple[GPT2WithARB, object, Config, Path]:
    """Build the model and load ARB weights from a checkpoint."""
    config = load_config(config_path)
    resolved_checkpoint = resolve_checkpoint_path(config, checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2WithARB(config)
    model.build_token_digit_tables(tokenizer)
    ckpt = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
    for key, state in ckpt["arb_state"].items():
        model.arbs[key].load_state_dict(state, strict=False)
    if "lora_state" in ckpt and model.lora_head is not None:
        model.lora_head.load_state_dict(ckpt["lora_state"])
    if "lora_layers_state" in ckpt and model.lora_layers is not None:
        model.lora_layers.load_state_dict(ckpt["lora_layers_state"])

    model.to(device)
    model.eval()
    return model, tokenizer, config, resolved_checkpoint


def find_operator_token(tokenizer, prompt: str) -> int | None:
    """Find the token index of the arithmetic operator in the prompt."""
    match = _PROMPT_PATTERN.match(prompt)
    if not match:
        return None
    # Tokenize the prefix up to and including the operator
    op_start = match.start(2)
    op_end = match.end(2)
    prefix = prompt[:op_end]
    prefix_ids = tokenizer.encode(prefix)
    # The operator token is the last token of the prefix
    return len(prefix_ids) - 1


def analyze_prompt(
    model: GPT2WithARB,
    tokenizer,
    device: torch.device,
    prompt: str,
) -> tuple[list[LayerExtraction], int, int]:
    """Run a forward pass on the prompt and decode extraction outputs."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    eq_index = int(attention_mask[0].sum().item()) - 1
    op_index = find_operator_token(tokenizer, prompt)
    extractions = collect_layer_extractions(
        outputs.get("arb_extractions", {}), eq_index
    )
    return extractions, op_index, eq_index


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


def print_result(
    checkpoint_path: Path,
    prompt: str,
    extractions: list[LayerExtraction],
    completion: str,
    full_text: str,
    op_index: int | None = None,
    eq_index: int | None = None,
) -> None:
    """Print extraction details alongside the usual inference output."""
    expected = compute_expected(prompt)

    print(f"checkpoint: {checkpoint_path}")
    print(f"prompt:     {prompt}")
    print(format_extractions(extractions, op_index, eq_index))
    print(f"completion: {completion.strip() or '<empty>'}")
    print(f"full_text:  {full_text}")
    if expected is not None:
        print(f"expected:   {expected}")
    print()


def interactive_loop(
    model: GPT2WithARB,
    tokenizer,
    device: torch.device,
    checkpoint_path: Path,
    max_new_tokens: int,
) -> None:
    """Run a simple prompt REPL."""
    print(
        "Enter a prompt like '347 * 291 ='. "
        "Press Ctrl-D or submit an empty line to exit.\n"
    )

    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            break

        if not prompt:
            break

        extractions, op_idx, eq_idx = analyze_prompt(model, tokenizer, device, prompt)
        full_text, completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        print_result(checkpoint_path, prompt, extractions, completion, full_text, op_idx, eq_idx)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prompt debugging directly against an ARB checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the training config used to build the model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to an ARB checkpoint. Defaults to the latest checkpoint.",
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

    device = get_device(args.device)
    logger.info("Using device: %s", device)
    model, tokenizer, _config, checkpoint_path = load_checkpointed_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    logger.info("Loaded checkpoint from: %s", checkpoint_path)

    if args.prompt is not None:
        extractions, op_idx, eq_idx = analyze_prompt(model, tokenizer, device, args.prompt)
        full_text, completion = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        print_result(checkpoint_path, args.prompt, extractions, completion, full_text, op_idx, eq_idx)
        return

    interactive_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        checkpoint_path=checkpoint_path,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
