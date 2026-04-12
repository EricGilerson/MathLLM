"""Tests for evaluation batching helpers."""

from __future__ import annotations

import math

import torch

from mathllm.config import EvalConfig
from mathllm.evaluation.evaluator import ARBEvaluator


class FakeTokenizer:
    """Tokenizer stub that maps characters directly to token ids."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"

    def __call__(
        self,
        texts,
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ):
        if isinstance(texts, str):
            texts = [texts]

        encoded = []
        for text in texts:
            token_ids = [ord(ch) for ch in text]
            if truncation and max_length is not None:
                token_ids = token_ids[:max_length]
            encoded.append(token_ids)

        max_len = max(len(token_ids) for token_ids in encoded) if encoded else 0
        if not padding:
            max_len = max(len(token_ids) for token_ids in encoded)

        padded = []
        attention_masks = []
        for token_ids in encoded:
            pad_len = max_len - len(token_ids)
            padded.append(token_ids + [0] * pad_len)
            attention_masks.append([1] * len(token_ids) + [0] * pad_len)

        if return_tensors != "pt":
            raise ValueError("FakeTokenizer only supports return_tensors='pt'")

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        chars = []
        for token_id in token_ids:
            if token_id == 0 and skip_special_tokens:
                continue
            chars.append(chr(token_id))
        return "".join(chars)


class FakeModel:
    """Model stub that records batching behavior."""

    def __init__(self):
        self.eval_calls = 0
        self.to_calls: list[torch.device] = []
        self.prepare_calls: list[torch.device] = []
        self.generate_calls: list[dict[str, int | tuple[int, int]]] = []
        self.forward_calls: list[dict[str, torch.Tensor]] = []

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.to_calls.append(torch.device(device))
        return self

    def prepare_for_device(self, device):
        self.prepare_calls.append(torch.device(device))

    def generate(self, input_ids, max_new_tokens=20, greedy=True):
        self.generate_calls.append(
            {
                "shape": tuple(input_ids.shape),
                "max_new_tokens": int(max_new_tokens),
            }
        )
        suffixes = []
        for row in input_ids:
            fill_token = row[0].item()
            suffixes.append(
                torch.full(
                    (max_new_tokens,),
                    fill_token,
                    dtype=row.dtype,
                    device=row.device,
                )
            )
        generated_suffix = torch.stack(suffixes, dim=0)
        return torch.cat([input_ids, generated_suffix], dim=1)

    def __call__(self, input_ids, attention_mask, labels):
        self.forward_calls.append(
            {
                "input_ids": input_ids.detach().cpu(),
                "attention_mask": attention_mask.detach().cpu(),
                "labels": labels.detach().cpu(),
            }
        )
        return {"loss": torch.tensor(2.0, device=input_ids.device)}


class TestEvaluatorBatching:
    def test_generate_texts_batches_same_shape_prompts(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        config = EvalConfig(batch_size=2, max_new_tokens=4)
        evaluator = ARBEvaluator(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=torch.device("cpu"),
        )

        completions = evaluator._generate_texts(
            ["1+1=", "2+2=", "10+10=", "3+3="],
            max_new_tokens=[2, 2, 3, 2],
        )

        assert completions == ["11", "22", "111", "33"]
        assert model.to_calls == [torch.device("cpu")]
        assert model.prepare_calls == [torch.device("cpu")]
        assert model.generate_calls == [
            {"shape": (2, 4), "max_new_tokens": 2},
            {"shape": (1, 4), "max_new_tokens": 2},
            {"shape": (1, 6), "max_new_tokens": 3},
        ]

    def test_perplexity_test_batches_and_masks_padding(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        config = EvalConfig(batch_size=2)
        evaluator = ARBEvaluator(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=torch.device("cpu"),
        )

        results = evaluator.perplexity_test(["ab", "wxyz", "k"], max_samples=3)

        assert results["avg_loss"] == 2.0
        assert math.isclose(results["perplexity"], math.exp(2.0), rel_tol=1e-6)
        assert len(model.forward_calls) == 1

        forward_call = model.forward_calls[0]
        torch.testing.assert_close(
            forward_call["attention_mask"],
            torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long),
        )
        torch.testing.assert_close(
            forward_call["labels"],
            torch.tensor([[97, 98, -100, -100], [119, 120, 121, 122]], dtype=torch.long),
        )
