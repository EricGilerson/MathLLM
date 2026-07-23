"""Small deterministic character tokenizer for local toy-pretraining runs."""

from __future__ import annotations

import torch


class CharTokenizer:
    """ASCII character tokenizer with standalone arithmetic symbols and digits."""

    _SPECIAL = ("<pad>", "<eos>", "<unk>")

    def __init__(self):
        chars = ["\n"] + [chr(code) for code in range(32, 127)]
        self.tokens = list(self._SPECIAL) + chars
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        ids = [self.token_to_id.get(char, self.unk_token_id) for char in text]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if return_tensors is None:
            return ids
        if return_tensors != "pt":
            raise ValueError("CharTokenizer supports return_tensors='pt' only")
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        special = set(self._SPECIAL)
        return "".join(
            "" if skip_special_tokens and self.id_to_token.get(int(token_id), self.unk_token) in special
            else self.id_to_token.get(int(token_id), self.unk_token)
            for token_id in ids
        )

    def __call__(self, texts, *, truncation=False, padding=False, max_length=None, return_tensors="pt", add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        target_length = max(len(ids) for ids in encoded) if encoded else 0
        if padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required for padding='max_length'")
            target_length = max_length
        elif padding:
            target_length = max(len(ids) for ids in encoded)
        input_ids = [ids + [self.pad_token_id] * max(0, target_length - len(ids)) for ids in encoded]
        attention_mask = [[1] * min(len(ids), target_length) + [0] * max(0, target_length - len(ids)) for ids in encoded]
        if return_tensors != "pt":
            raise ValueError("CharTokenizer supports return_tensors='pt' only")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
