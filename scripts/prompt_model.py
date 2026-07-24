#!/usr/bin/env python3
"""Prompt an exported 360M model, a raw Hugging Face model, or a scratch checkpoint.

Examples:
  python scripts/prompt_model.py --target exported-arb --model-dir trained_model_360m
  python scripts/prompt_model.py --target foundation --prompt '35/7='
  python scripts/prompt_model.py --target toy-arb --toy-config configs/toy_pretrain_gpu_seed1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.model.utils import get_device
from mathllm.pretraining.arithmetic_bpe_tokenizer import ArithmeticBPETokenizer
from mathllm.pretraining.experiment import build_model, load_toy_config


def _load_exported_arb(model_dir: Path, device: torch.device):
    model, tokenizer, _ = GPT2WithARB.from_exported_model(model_dir, device=device)
    return model.eval(), tokenizer, True


def _load_huggingface(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, False


def _load_toy(variant: str, config_path: Path, checkpoint_path: Path | None, device: torch.device):
    config = load_toy_config(config_path)
    tokenizer_path = Path(config.data.tokenizer_file)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Toy tokenizer not found: {tokenizer_path}. Run prepare_toy_pretraining_data first."
        )
    if checkpoint_path is None:
        checkpoint_path = Path(config.training.output_dir) / variant / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Toy {variant} checkpoint not found: {checkpoint_path}")

    tokenizer = ArithmeticBPETokenizer.from_file(tokenizer_path)
    model = build_model(config, variant, tokenizer)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), tokenizer, variant == "arb"


def _generate(model, tokenizer, is_arb: bool, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        if is_arb:
            output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, greedy=True)
        else:
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("exported-arb", "foundation", "instruct", "toy-baseline", "toy-arb"),
        default="exported-arb",
        help="Which checkpoint family to load.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("trained_model_360m"), help="Exported ARB bundle.")
    parser.add_argument("--foundation-model", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--instruct-model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--toy-config", type=Path, default=None, help="Seed-specific toy-pretraining YAML.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional toy checkpoint override.")
    parser.add_argument("--prompt", default=None, help="Run one prompt; omit for an interactive REPL.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()

    device = get_device(args.device)
    if args.target == "exported-arb":
        model, tokenizer, is_arb = _load_exported_arb(args.model_dir, device)
    elif args.target == "foundation":
        model, tokenizer, is_arb = _load_huggingface(args.foundation_model, device)
    elif args.target == "instruct":
        model, tokenizer, is_arb = _load_huggingface(args.instruct_model, device)
    else:
        if args.toy_config is None:
            parser.error("--toy-config is required for toy-baseline and toy-arb")
        variant = "baseline" if args.target == "toy-baseline" else "arb"
        model, tokenizer, is_arb = _load_toy(variant, args.toy_config, args.checkpoint, device)

    def answer(prompt: str) -> None:
        full_text = _generate(model, tokenizer, is_arb, prompt, args.max_new_tokens, device)
        print(f"\nprompt:     {prompt}\ncompletion: {full_text[len(prompt):]}\nfull_text:  {full_text}\n")

    if args.prompt is not None:
        answer(args.prompt)
        return

    print(f"Loaded {args.target} on {device}. Enter a prompt; blank line or Ctrl-D exits.")
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            print()
            break
        if not prompt.strip():
            break
        answer(prompt)


if __name__ == "__main__":
    main()
