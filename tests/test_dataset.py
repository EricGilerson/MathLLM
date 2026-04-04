"""Tests for dataset label masking behavior."""

from __future__ import annotations

import torch

from mathllm.data.dataset import ArithmeticDataset, _infer_target_start


class CharTokenizer:
    """Simple character-level tokenizer for deterministic tests."""

    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(
        self,
        text,
        truncation: bool = True,
        padding: str | None = None,
        max_length: int = 128,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
    ):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        tokenized = []
        attention_masks = []
        for item in texts:
            ids = [ord(ch) for ch in item]
            if truncation:
                ids = ids[:max_length]
            mask = [1] * len(ids)
            if padding == "max_length":
                pad_len = max_length - len(ids)
                ids = ids + [0] * pad_len
                mask = mask + [0] * pad_len
            tokenized.append(ids)
            attention_masks.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(tokenized, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

        return {
            "input_ids": tokenized[0] if len(tokenized) == 1 else tokenized,
            "attention_mask": attention_masks[0] if len(attention_masks) == 1 else attention_masks,
        }

    def encode(self, text: str, return_tensors: str | None = None):
        encoded = [ord(ch) for ch in text]
        if return_tensors == "pt":
            return torch.tensor([encoded], dtype=torch.long)
        return encoded


def test_infer_target_start_uses_last_numeric_value():
    text = "347 * 291 = 100977"
    assert _infer_target_start(text) == text.rfind("100977")


def test_answer_only_loss_masks_prompt_tokens():
    tokenizer = CharTokenizer()
    text = "347 * 291 = 100977"
    ds = ArithmeticDataset(
        examples=[text],
        tokenizer=tokenizer,
        max_length=64,
        answer_only_loss=True,
    )

    item = ds[0]
    target_start = text.rfind("100977")
    assert torch.all(item["labels"][:target_start] == -100)
    assert torch.equal(
        item["labels"][target_start:target_start + len("100977")],
        item["input_ids"][target_start:target_start + len("100977")],
    )


def test_non_arithmetic_examples_keep_full_loss():
    tokenizer = CharTokenizer()
    text = "Flight 347 departs from gate 12."
    ds = ArithmeticDataset(
        examples=[text],
        tokenizer=tokenizer,
        max_length=64,
        answer_only_loss=True,
    )

    item = ds[0]
    valid = item["attention_mask"].bool()
    assert torch.equal(item["labels"][valid], item["input_ids"][valid])


def test_explicit_target_start_overrides_inference():
    tokenizer = CharTokenizer()
    text = "Answer: 100977"
    ds = ArithmeticDataset(
        examples=[{"text": text, "target_start": text.index("100977")}],
        tokenizer=tokenizer,
        max_length=64,
        answer_only_loss=True,
    )

    item = ds[0]
    target_start = text.index("100977")
    assert torch.all(item["labels"][:target_start] == -100)
