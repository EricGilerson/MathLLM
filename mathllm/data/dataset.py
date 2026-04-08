"""PyTorch Dataset wrappers for arithmetic training data."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ARITHMETIC_PATTERNS = (
    re.compile(r"\d[\d,]*(?:\.\d+)?\s*[+\-*/×÷^]\s*-?\d[\d,]*(?:\.\d+)?"),
    re.compile(r"\b(?:plus|minus|times|multiplied by|divided by|sum of|difference|product|quotient)\b", re.IGNORECASE),
    re.compile(r"\b(?:sin|cos|tan|sqrt|ln|exp)\s*\(", re.IGNORECASE),
)


def _looks_like_arithmetic(text: str) -> bool:
    """Return True when text resembles a supervised arithmetic example."""
    return any(pattern.search(text) for pattern in _ARITHMETIC_PATTERNS)


def _infer_target_start(text: str) -> int | None:
    """Infer the start index of the answer span for arithmetic examples.

    The generated templates consistently place the final answer as the last
    numeric value in the example. For arithmetic examples we therefore mask
    the prompt and train only on that final numeric suffix.
    """
    if not _looks_like_arithmetic(text):
        return None

    matches = list(_NUMBER_PATTERN.finditer(text))
    if not matches:
        return None
    return matches[-1].start()


def _int_to_digits(n: int | None, num_digits: int = 10) -> list[int]:
    """Decompose integer to base-10 digit vector, LSB-first, zero-padded."""
    if n is None:
        return [0] * num_digits
    n = abs(n)
    return [(n // 10**k) % 10 for k in range(num_digits)]


def _normalize_example(
    raw: str | dict[str, Any],
    *,
    answer_only_loss: bool,
) -> dict[str, Any]:
    """Normalize legacy and structured example records."""
    # Handle ArithmeticRecord dataclass objects
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()

    if isinstance(raw, str):
        text = raw
        target_start = _infer_target_start(text) if answer_only_loss else None
        return {
            "text": text,
            "target_start": target_start,
            "operand_a": None,
            "operand_b": None,
            "op_type": "unknown",
        }

    text = str(raw["text"])
    target_start = raw.get("target_start")
    if target_start is None and answer_only_loss:
        target_start = _infer_target_start(text)
    return {
        "text": text,
        "target_start": target_start,
        "operand_a": raw.get("operand_a"),
        "operand_b": raw.get("operand_b"),
        "op_type": raw.get("op_type", "unknown"),
    }


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
        examples: list[str | dict[str, Any]] | None = None,
        jsonl_path: str | Path | None = None,
        tokenizer=None,
        max_length: int = 128,
        augment: bool = False,
        seed: int = 42,
        answer_only_loss: bool = False,
        num_digits: int = 10,
    ):
        """
        Args:
            examples: List of text examples (mutually exclusive with jsonl_path)
            jsonl_path: Path to JSONL file with {"text": ...} lines
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            augment: Apply random formatting augmentations at access time
            seed: Random seed for augmentation
            num_digits: Number of digit slots for auxiliary targets
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
        self.answer_only_loss = answer_only_loss
        self.num_digits = num_digits
        self.records = [
            _normalize_example(example, answer_only_loss=answer_only_loss)
            for example in self.examples
        ]
        self.examples = [record["text"] for record in self.records]

        # Pre-compute auxiliary digit vectors for extraction loss
        self._build_aux_targets()

        # Tokenize all examples upfront for efficiency (when not augmenting)
        if tokenizer is not None and not augment:
            self._tokenize()
        else:
            self.encodings = None
            if tokenizer is not None and augment:
                # Ensure pad token exists for per-item tokenization
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_aux_targets(self) -> None:
        """Pre-compute ground-truth digit vectors for auxiliary extraction loss."""
        n = len(self.records)
        self.digits_a = torch.zeros(n, self.num_digits, dtype=torch.float32)
        self.digits_b = torch.zeros(n, self.num_digits, dtype=torch.float32)
        self.has_aux = torch.zeros(n, dtype=torch.bool)
        self.max_operand_digits = torch.zeros(n, dtype=torch.long)
        # eq_positions will be populated after tokenization
        self.eq_positions = torch.zeros(n, dtype=torch.long)

        for i, rec in enumerate(self.records):
            op_a = rec.get("operand_a")
            op_b = rec.get("operand_b")
            if op_a is not None and op_b is not None:
                self.digits_a[i] = torch.tensor(
                    _int_to_digits(op_a, self.num_digits), dtype=torch.float32
                )
                self.digits_b[i] = torch.tensor(
                    _int_to_digits(op_b, self.num_digits), dtype=torch.float32
                )
                self.has_aux[i] = True
            # Compute max digit count for curriculum filtering
            d = 0
            if op_a is not None:
                d = max(d, len(str(abs(op_a))))
            if op_b is not None:
                d = max(d, len(str(abs(op_b))))
            self.max_operand_digits[i] = d

    @staticmethod
    def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
        """Load text examples from JSONL file."""
        examples = []
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(data)
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
        self.labels = self.encodings["input_ids"].clone()
        self.labels[self.encodings["attention_mask"] == 0] = -100

        for idx, record in enumerate(self.records):
            target_start = record["target_start"]
            # Compute eq_position: token index where both operands are visible.
            # We use target_start (char offset of the answer) as a proxy — by
            # the answer position, both operands have been seen via causal attn.
            # For examples without target_start, infer it for eq_position only.
            eq_char = target_start
            if eq_char is None:
                eq_char = _infer_target_start(record["text"])
            if eq_char is not None:
                eq_prefix = record["text"][:eq_char]
                eq_prefix_ids = self.tokenizer(
                    eq_prefix,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                )["input_ids"]
                # Point to the last prefix token (the '=' sign), not one past it.
                # This is the position where both operands are visible but the
                # answer has not yet appeared — the cleanest extraction point.
                self.eq_positions[idx] = max(0, min(len(eq_prefix_ids) - 1, self.max_length - 1))

            if target_start is None:
                continue

            prefix = record["text"][:target_start]
            prefix_ids = self.tokenizer(
                prefix,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"]
            prefix_len = min(len(prefix_ids), self.max_length)
            self.labels[idx, :prefix_len] = -100

    def set_tokenizer(self, tokenizer) -> None:
        """Set tokenizer and tokenize examples (also computes eq_positions)."""
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
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # Compute eq_position dynamically for augmented text
            eq_char = _infer_target_start(text)
            if eq_char is not None:
                eq_prefix_ids = self.tokenizer(
                    text[:eq_char],
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                )["input_ids"]
                eq_position = torch.tensor(
                    max(0, min(len(eq_prefix_ids) - 1, self.max_length - 1)), dtype=torch.long
                )
            else:
                eq_position = torch.tensor(0, dtype=torch.long)

            target_start = eq_char if self.answer_only_loss else None
            if target_start is not None:
                prefix = text[:target_start]
                prefix_ids = self.tokenizer(
                    prefix,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                )["input_ids"]
                prefix_len = min(len(prefix_ids), self.max_length)
                labels[:prefix_len] = -100
        else:
            if self.encodings is None:
                raise RuntimeError("Tokenizer not set. Call set_tokenizer() first.")
            input_ids = self.encodings["input_ids"][idx]
            attention_mask = self.encodings["attention_mask"][idx]
            labels = self.labels[idx]
            eq_position = self.eq_positions[idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "digits_a": self.digits_a[idx],
            "digits_b": self.digits_b[idx],
            "has_aux": self.has_aux[idx],
            "eq_position": eq_position,
        }

    def split(self, train_ratio: float = 0.9) -> tuple["ArithmeticDataset", "ArithmeticDataset"]:
        """Split into train and eval datasets."""
        n = len(self.examples)
        split_idx = int(n * train_ratio)

        train_ds = ArithmeticDataset(
            examples=self.records[:split_idx],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            augment=self.augment,
            answer_only_loss=self.answer_only_loss,
            num_digits=self.num_digits,
        )
        # Never augment eval data — we want deterministic evaluation
        eval_ds = ArithmeticDataset(
            examples=self.records[split_idx:],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            augment=False,
            answer_only_loss=self.answer_only_loss,
            num_digits=self.num_digits,
        )
        return train_ds, eval_ds
