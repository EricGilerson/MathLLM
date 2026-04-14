"""Tests for evaluation batching helpers."""

from __future__ import annotations

import math
import re

import torch

from mathllm.config import EvalConfig
from mathllm.evaluation import evaluator as evaluator_module
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


class DeterministicEvaluator(ARBEvaluator):
    """Evaluator stub that returns the mathematically correct integer answer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_history: list[str] = []

    def _prepare_model(self) -> None:
        return None

    def _generate_texts(self, prompts, max_new_tokens=None):
        self.prompt_history.extend(prompts)
        return [self._solve_prompt(prompt) for prompt in prompts]

    @staticmethod
    def _solve_prompt(prompt: str) -> str:
        expression = prompt.rstrip("=")
        if "^" in expression:
            left, right = expression.split("^")
            return str(int(left) ** int(right))

        match = re.fullmatch(r"(\d+)([+\-*/])(\d+)", expression)
        if match is None:
            raise AssertionError(f"Unexpected prompt format: {prompt}")

        left = int(match.group(1))
        op = match.group(2)
        right = int(match.group(3))
        if op == "+":
            return str(left + right)
        if op == "-":
            return str(left - right)
        if op == "*":
            return str(left * right)
        if op == "/":
            return str(left // right)
        raise AssertionError(f"Unsupported prompt format: {prompt}")


class FullEvaluationStub(ARBEvaluator):
    """Evaluator stub that records which suite methods are called."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.called: list[str] = []

    def exact_match_accuracy(self, num_samples=None, seed=12345):
        self.called.append("exact_match")
        return {"ok": 1.0}

    def division_accuracy(self, num_samples=None, seed=54321):
        self.called.append("division")
        return {"ok": 1.0}

    def transcendental_accuracy(self, num_samples=None, tolerance=1e-4, seed=67890):
        raise AssertionError("transcendental_accuracy should not be called")

    def float_arithmetic_accuracy(self, num_samples=None, tolerance=1e-4, seed=11111):
        raise AssertionError("float_arithmetic_accuracy should not be called")

    def multi_step_accuracy(self, num_samples=None, seed=22222):
        raise AssertionError("multi_step_accuracy should not be called")

    def ablation_test(self, num_samples=50, seed=99):
        raise AssertionError("ablation_test should not be called")

    def four_way_ablation(self, num_samples=50, seed=99):
        raise AssertionError("four_way_ablation should not be called")

    def perplexity_test(self, eval_texts, max_samples=200):
        raise AssertionError("perplexity_test should not be called")


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

    def test_generate_texts_caps_requested_tokens_to_eval_config(self):
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
            ["1+1=", "2+2="],
            max_new_tokens=[2, 10],
        )

        assert completions == ["11", "2222"]
        assert model.generate_calls == [
            {"shape": (1, 4), "max_new_tokens": 2},
            {"shape": (1, 4), "max_new_tokens": 4},
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


class TestEvaluatorDigitPairs:
    def test_exact_match_reports_mixed_digit_pairs_and_legacy_diagonals(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        config = EvalConfig(num_samples_per_config=1, max_digits_range=(1, 2))
        evaluator = DeterministicEvaluator(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=torch.device("cpu"),
        )

        results = evaluator.exact_match_accuracy(num_samples=1)

        expected_pairs = {"1x1", "1x2", "2x1", "2x2"}
        for op in ("add", "sub", "mul"):
            pair_results = results[f"{op}_digit_pairs"]
            assert set(pair_results) == expected_pairs
            assert all(accuracy == 1.0 for accuracy in pair_results.values())
            assert results[f"{op}_1digit"] == pair_results["1x1"]
            assert results[f"{op}_2digit"] == pair_results["2x2"]
            assert results[f"{op}_overall_mean"] == 1.0
            assert results[f"{op}_diagonal_mean"] == 1.0
            assert results[f"{op}_cross_digit_mean"] == 1.0

    def test_subtraction_keeps_ordered_pair_entries_when_operands_swap(self, monkeypatch):
        def fake_sample_number(num_digits, rng, min_value=0):
            if num_digits == 1:
                return max(min_value, 0)
            return max(min_value, 10 ** (num_digits - 1))

        monkeypatch.setattr(evaluator_module, "_sample_number", fake_sample_number)

        model = FakeModel()
        tokenizer = FakeTokenizer()
        config = EvalConfig(num_samples_per_config=1, max_digits_range=(1, 3))
        evaluator = DeterministicEvaluator(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=torch.device("cpu"),
        )

        results = evaluator.exact_match_accuracy(num_samples=1)

        assert results["sub_digit_pairs"]["1x3"] == 1.0
        assert results["sub_digit_pairs"]["3x1"] == 1.0

    def test_division_reports_mixed_digit_pairs_and_legacy_diagonals(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        config = EvalConfig(num_samples_per_config=1, max_digits_range=(1, 2))
        evaluator = DeterministicEvaluator(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=torch.device("cpu"),
        )

        results = evaluator.division_accuracy(num_samples=1)

        expected_pairs = {"1x1", "1x2", "2x1", "2x2"}
        assert set(results["div_digit_pairs"]) == expected_pairs
        assert all(accuracy == 1.0 for accuracy in results["div_digit_pairs"].values())
        assert results["div_1digit"] == results["div_digit_pairs"]["1x1"]
        assert results["div_2digit"] == results["div_digit_pairs"]["2x2"]
        assert results["div_overall_mean"] == 1.0
        assert results["div_diagonal_mean"] == 1.0
        assert results["div_cross_digit_mean"] == 1.0


class TestFullEvaluation:
    def test_full_evaluation_only_runs_integer_suites(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        config = EvalConfig()
        evaluator = FullEvaluationStub(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=torch.device("cpu"),
        )

        results = evaluator.full_evaluation(eval_texts=["language sample"])

        assert results == {
            "exact_match": {"ok": 1.0},
            "division": {"ok": 1.0},
        }
        assert evaluator.called == ["exact_match", "division"]
