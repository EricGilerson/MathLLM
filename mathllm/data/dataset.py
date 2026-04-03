"""PyTorch Dataset wrappers for arithmetic training data."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _augment_text(text: str, rng: random.Random) -> str:
    """Apply random formatting variations to a text example.

    Augmentations are lightweight and preserve mathematical correctness:
    - Random whitespace around operators (= + - * /)
    - Occasional comma insertion in large numbers
    - Random case changes in non-number text
    """
    # 30% chance: vary whitespace around operators
    if rng.random() < 0.3:
        def _vary_op_whitespace(match: re.Match) -> str:
            op = match.group(0).strip()
            style = rng.choice(["tight", "spaced", "left", "right"])
            if style == "tight":
                return op
            elif style == "spaced":
                return f" {op} "
            elif style == "left":
                return f" {op}"
            else:
                return f"{op} "
        text = re.sub(r'\s*([=+\-*/×÷^])\s*', _vary_op_whitespace, text)

    # 20% chance: insert commas into large numbers (>= 1000)
    if rng.random() < 0.2:
        def _comma_number(match: re.Match) -> str:
            num_str = match.group(0)
            if len(num_str) >= 4 and rng.random() < 0.5:
                n = int(num_str)
                return f"{n:,}"
            return num_str
        text = re.sub(r'\b\d{4,}\b', _comma_number, text)

    # 15% chance: random case variation on a word
    if rng.random() < 0.15:
        words = text.split()
        if len(words) > 2:
            idx = rng.randint(0, len(words) - 1)
            # Only modify non-number words
            if not words[idx].replace(',', '').replace('.', '').isdigit():
                case = rng.choice(["upper", "lower", "title"])
                if case == "upper":
                    words[idx] = words[idx].upper()
                elif case == "lower":
                    words[idx] = words[idx].lower()
                else:
                    words[idx] = words[idx].title()
                text = " ".join(words)

    return text


class ArithmeticDataset(Dataset):
    """PyTorch Dataset for tokenized arithmetic examples.

    Loads text examples from a JSONL file, tokenizes them, and returns
    input_ids / attention_mask / labels for causal language modeling.

    When augment=True, applies random formatting variations at access time
    so the model sees slightly different text each epoch. This requires
    per-item tokenization (slower but more robust to overfitting).
    """

    def __init__(
        self,
        examples: list[str] | None = None,
        jsonl_path: str | Path | None = None,
        tokenizer=None,
        max_length: int = 128,
        augment: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            examples: List of text examples (mutually exclusive with jsonl_path)
            jsonl_path: Path to JSONL file with {"text": ...} lines
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            augment: Apply random formatting augmentations at access time
            seed: Random seed for augmentation
        """
        if examples is None and jsonl_path is not None:
            examples = self._load_jsonl(jsonl_path)
        elif examples is None:
            raise ValueError("Either examples or jsonl_path must be provided")

        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.aug_rng = random.Random(seed)

        # Tokenize all examples upfront for efficiency (when not augmenting)
        if tokenizer is not None and not augment:
            self._tokenize()
        else:
            self.encodings = None
            if tokenizer is not None and augment:
                # Ensure pad token exists for per-item tokenization
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _load_jsonl(path: str | Path) -> list[str]:
        """Load text examples from JSONL file."""
        examples = []
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(data["text"])
        return examples

    def _tokenize(self) -> None:
        """Tokenize all examples."""
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.encodings = self.tokenizer(
            self.examples,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def set_tokenizer(self, tokenizer) -> None:
        """Set tokenizer and tokenize examples."""
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not self.augment:
            self._tokenize()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set. Call set_tokenizer() first.")

        if self.augment:
            # Per-item tokenization with augmentation
            text = _augment_text(self.examples[idx], self.aug_rng)
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            if self.encodings is None:
                raise RuntimeError("Tokenizer not set. Call set_tokenizer() first.")
            input_ids = self.encodings["input_ids"][idx]
            attention_mask = self.encodings["attention_mask"][idx]

        # Labels: same as input_ids for causal LM.
        # Set padding positions to -100 so they're ignored in loss.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def split(self, train_ratio: float = 0.9) -> tuple["ArithmeticDataset", "ArithmeticDataset"]:
        """Split into train and eval datasets."""
        n = len(self.examples)
        split_idx = int(n * train_ratio)

        train_ds = ArithmeticDataset(
            examples=self.examples[:split_idx],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            augment=self.augment,
        )
        # Never augment eval data — we want deterministic evaluation
        eval_ds = ArithmeticDataset(
            examples=self.examples[split_idx:],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            augment=False,
        )
        return train_ds, eval_ds
