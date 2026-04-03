"""PyTorch Dataset wrappers for arithmetic training data."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ArithmeticDataset(Dataset):
    """PyTorch Dataset for tokenized arithmetic examples.

    Loads text examples from a JSONL file, tokenizes them, and returns
    input_ids / attention_mask / labels for causal language modeling.
    """

    def __init__(
        self,
        examples: list[str] | None = None,
        jsonl_path: str | Path | None = None,
        tokenizer=None,
        max_length: int = 128,
    ):
        """
        Args:
            examples: List of text examples (mutually exclusive with jsonl_path)
            jsonl_path: Path to JSONL file with {"text": ...} lines
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        if examples is None and jsonl_path is not None:
            examples = self._load_jsonl(jsonl_path)
        elif examples is None:
            raise ValueError("Either examples or jsonl_path must be provided")

        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize all examples upfront for efficiency
        if tokenizer is not None:
            self._tokenize()
        else:
            self.encodings = None

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
        self._tokenize()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
        )
        eval_ds = ArithmeticDataset(
            examples=self.examples[split_idx:],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return train_ds, eval_ds
