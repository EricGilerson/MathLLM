"""Compact BPE tokenizer that preserves arithmetic syntax token by token.

Natural-language spans use byte-level BPE.  Digits and the five arithmetic
symbols are split before BPE encoding, so ``123+45=`` is always represented as
the seven literal character tokens required by ARB extraction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class ArithmeticBPETokenizer:
    """Byte-level BPE for prose with atomic digits and arithmetic symbols."""

    _SPECIAL = ("<pad>", "<eos>", "<unk>")
    _ARITHMETIC = tuple("0123456789+-*/=")
    _SPLIT = re.compile(r"([0-9+\-*/=])")

    def __init__(self, backend: Tokenizer):
        self.backend = backend
        self.tokens = list(self._SPECIAL) + list(self._ARITHMETIC)
        self.token_to_id = {
            token: self._required_id(token) for token in self.tokens
        }
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}
        self.pad_token, self.eos_token, self.unk_token = self._SPECIAL
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

    def _required_id(self, token: str) -> int:
        token_id = self.backend.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required token {token!r}")
        return token_id

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "ArithmeticBPETokenizer":
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(cls._SPECIAL) + list(cls._ARITHMETIC),
        )
        backend.train_from_iterator(texts, trainer=trainer)
        return cls(backend)

    @classmethod
    def from_file(cls, path: str | Path) -> "ArithmeticBPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        ids: list[int] = []
        for piece in self._SPLIT.split(text):
            if not piece:
                continue
            token_id = self.token_to_id.get(piece)
            if token_id is not None and len(piece) == 1:
                ids.append(token_id)
            else:
                ids.extend(self.backend.encode(piece).ids)
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if return_tensors is None:
            return ids
        if return_tensors != "pt":
            raise ValueError("ArithmeticBPETokenizer supports return_tensors='pt' only")
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        result: list[str] = []
        bpe_ids: list[int] = []

        def flush_bpe() -> None:
            if bpe_ids:
                result.append(self.backend.decode(bpe_ids, skip_special_tokens=False))
                bpe_ids.clear()

        for token_id in map(int, ids):
            token = self.id_to_token.get(token_id)
            if token in self._ARITHMETIC:
                flush_bpe()
                result.append(token)
            elif token in self._SPECIAL:
                flush_bpe()
                if not skip_special_tokens:
                    result.append(token)
            else:
                bpe_ids.append(token_id)
        flush_bpe()
        return "".join(result)

    def __call__(self, texts, *, truncation=False, padding=False, max_length=None, return_tensors="pt", add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        target_length = max((len(ids) for ids in encoded), default=0)
        if padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required for padding='max_length'")
            target_length = max_length
        elif padding:
            target_length = max((len(ids) for ids in encoded), default=0)
        input_ids = [ids + [self.pad_token_id] * max(0, target_length - len(ids)) for ids in encoded]
        attention_mask = [[1] * min(len(ids), target_length) + [0] * max(0, target_length - len(ids)) for ids in encoded]
        if return_tensors != "pt":
            raise ValueError("ArithmeticBPETokenizer supports return_tensors='pt' only")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
