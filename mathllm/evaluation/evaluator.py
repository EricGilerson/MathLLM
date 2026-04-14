"""Evaluation suite for ARB-augmented models.

Measures:
1. Exact-match accuracy on N-digit arithmetic operations
2. Exact-match accuracy on exact division
3. Approximate accuracy on transcendental functions (sin, cos, tan, exp, ln, sqrt)
4. Approximate accuracy on floating-point arithmetic
5. Multi-step composition accuracy
6. Ablation: zero W_proj to confirm ARB is the source of improvement
7. Perplexity regression on language benchmarks
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mathllm.config import EvalConfig
from mathllm.model.gpt2_arb import GPT2WithARB

logger = logging.getLogger(__name__)

OP_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "exp": "^", "div": "/"}


def _compute_expected(op: str, a: int, b: int) -> int | None:
    """Compute the expected result for an operation."""
    if op == "add":
        return a + b
    elif op == "sub":
        return a - b
    elif op == "mul":
        return a * b
    elif op == "exp":
        try:
            result = a**b
            if result > 10**12:
                return None
            return result
        except (OverflowError, ValueError):
            return None
    elif op == "div":
        if b == 0 or a % b != 0:
            return None
        return a // b
    return None


def _sample_operands(num_digits: int, rng) -> tuple[int, int]:
    """Sample operands with specified digit count."""
    low = 10 ** (num_digits - 1) if num_digits > 1 else 0
    high = 10**num_digits - 1
    return rng.randint(low, high), rng.randint(low, high)


def _sample_number(num_digits: int, rng, min_value: int = 0) -> int:
    """Sample one integer with the requested digit count."""
    low = 10 ** (num_digits - 1) if num_digits > 1 else 0
    low = max(low, min_value)
    high = 10**num_digits - 1
    return rng.randint(low, high)


class ARBEvaluator:
    """Comprehensive evaluation of ARB-augmented models."""

    def __init__(
        self,
        model: GPT2WithARB,
        tokenizer,
        config: EvalConfig,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cpu")
        self.batch_size = max(1, int(getattr(config, "batch_size", 64)))
        self._prepared_device: torch.device | None = None

    def _prepare_model(self) -> None:
        """Move the model once and warm runtime caches for evaluation."""
        self.model.eval()
        if self._prepared_device == self.device:
            return

        self.model.to(self.device)
        if hasattr(self.model, "prepare_for_device"):
            self.model.prepare_for_device(self.device)
        self._prepared_device = self.device

    def _resolve_max_new_tokens(
        self,
        max_new_tokens: int | list[int] | None = None,
    ) -> int | list[int]:
        """Apply the configured generation cap to evaluation requests."""
        configured_max = max(1, int(self.config.max_new_tokens))
        if isinstance(max_new_tokens, list):
            return [max(1, min(int(tokens), configured_max)) for tokens in max_new_tokens]
        if max_new_tokens is None:
            return configured_max
        return max(1, min(int(max_new_tokens), configured_max))

    def _generate_texts(
        self,
        prompts: list[str],
        max_new_tokens: int | list[int] | None = None,
    ) -> list[str]:
        """Generate completions for a batch of prompts.

        The current model generation path assumes equal prompt lengths, so
        requests are grouped by tokenized length and generation length to
        batch safely without padding artifacts.
        """
        if not prompts:
            return []

        self._prepare_model()

        if isinstance(max_new_tokens, list):
            resolved_max_new_tokens = self._resolve_max_new_tokens(max_new_tokens)
            if len(resolved_max_new_tokens) != len(prompts):
                raise ValueError("max_new_tokens list must match prompts length")
            max_tokens_per_prompt = resolved_max_new_tokens
        else:
            max_tokens = self._resolve_max_new_tokens(max_new_tokens)
            max_tokens_per_prompt = [max_tokens] * len(prompts)

        grouped_requests: dict[tuple[int, int], list[tuple[int, str, torch.Tensor]]] = defaultdict(list)
        completions = [""] * len(prompts)

        for idx, (prompt, max_tokens) in enumerate(zip(prompts, max_tokens_per_prompt)):
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
            grouped_requests[(int(input_ids.numel()), int(max_tokens))].append(
                (idx, prompt, input_ids)
            )

        for (input_len, max_tokens), requests in grouped_requests.items():
            for start in range(0, len(requests), self.batch_size):
                batch = requests[start:start + self.batch_size]
                batch_input_ids = torch.stack(
                    [input_ids for _, _, input_ids in batch], dim=0
                ).to(self.device)

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        batch_input_ids,
                        max_new_tokens=max_tokens,
                        greedy=True,
                    )

                output_ids = output_ids[:, input_len:].detach().cpu()
                for row, (idx, _, _) in enumerate(batch):
                    completions[idx] = self.tokenizer.decode(
                        output_ids[row].tolist(),
                        skip_special_tokens=True,
                    )

        return completions

    def _generate_text(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Generate text from a prompt."""
        return self._generate_texts([prompt], max_new_tokens=max_new_tokens)[0]

    def _extract_number_from_generation(self, text: str) -> str | None:
        """Extract the first integer (possibly negative) from generated text."""
        text = text.strip()
        match = re.search(r"-?\d+", text)
        if match:
            return match.group()
        return None

    def _extract_float_from_generation(self, text: str) -> float | None:
        """Extract the first float (possibly negative) from generated text."""
        text = text.strip()
        match = re.search(r"-?\d+\.?\d*", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

    def _digit_pairs(self, max_digits: int | None = None) -> list[tuple[int, int]]:
        """Enumerate ordered digit-count pairs over the configured range."""
        min_d, configured_max = self.config.max_digits_range
        upper = configured_max if max_digits is None else min(configured_max, max_digits)
        return [
            (digits_a, digits_b)
            for digits_a in range(min_d, upper + 1)
            for digits_b in range(min_d, upper + 1)
        ]

    def _new_digit_error_counts(self) -> dict[str, object]:
        """Create an accumulator for digit-position error tracking."""
        return {
            "total_cases": 0,
            "wrong_cases": 0,
            "parsed_wrong_cases": 0,
            "unparsed_cases": 0,
            "sign_mismatch_cases": 0,
            "positions": {},
        }

    def _accumulate_digit_error_counts(
        self,
        counts: dict[str, object],
        expected: int,
        extracted: str | None,
    ) -> None:
        """Track where integer generations go wrong using left-to-right positions."""
        counts["total_cases"] += 1

        if extracted is None:
            counts["wrong_cases"] += 1
            counts["unparsed_cases"] += 1
            return

        parsed_value = int(extracted)
        if parsed_value != expected:
            counts["wrong_cases"] += 1
            counts["parsed_wrong_cases"] += 1

        expected_negative = expected < 0
        extracted_negative = extracted.startswith("-")
        if expected_negative != extracted_negative:
            counts["sign_mismatch_cases"] += 1

        expected_digits = str(abs(expected))
        extracted_digits = extracted.lstrip("-")
        max_positions = max(len(expected_digits), len(extracted_digits))
        position_counts: dict[int, dict[str, int]] = counts["positions"]  # type: ignore[assignment]

        for position in range(max_positions):
            expected_digit = expected_digits[position] if position < len(expected_digits) else None
            extracted_digit = extracted_digits[position] if position < len(extracted_digits) else None
            bucket = position_counts.setdefault(position + 1, {"evaluated": 0, "wrong": 0})
            bucket["evaluated"] += 1
            if expected_digit != extracted_digit:
                bucket["wrong"] += 1

    def _merge_digit_error_counts(
        self,
        left: dict[str, object],
        right: dict[str, object],
    ) -> dict[str, object]:
        """Merge two digit-position error accumulators."""
        left["total_cases"] += right["total_cases"]
        left["wrong_cases"] += right["wrong_cases"]
        left["parsed_wrong_cases"] += right["parsed_wrong_cases"]
        left["unparsed_cases"] += right["unparsed_cases"]
        left["sign_mismatch_cases"] += right["sign_mismatch_cases"]

        left_positions: dict[int, dict[str, int]] = left["positions"]  # type: ignore[assignment]
        right_positions: dict[int, dict[str, int]] = right["positions"]  # type: ignore[assignment]
        for position, stats in right_positions.items():
            bucket = left_positions.setdefault(position, {"evaluated": 0, "wrong": 0})
            bucket["evaluated"] += stats["evaluated"]
            bucket["wrong"] += stats["wrong"]
        return left

    def _finalize_digit_error_profile(
        self,
        counts: dict[str, object],
    ) -> dict[str, object]:
        """Convert digit-position error counts into reportable metrics."""
        total_cases = int(counts["total_cases"])
        wrong_cases = int(counts["wrong_cases"])
        parsed_wrong_cases = int(counts["parsed_wrong_cases"])
        unparsed_cases = int(counts["unparsed_cases"])
        sign_mismatch_cases = int(counts["sign_mismatch_cases"])
        raw_positions: dict[int, dict[str, int]] = counts["positions"]  # type: ignore[assignment]

        wrong_positions_total = sum(stats["wrong"] for stats in raw_positions.values())
        positions: dict[str, object] = {}
        for position in sorted(raw_positions):
            stats = raw_positions[position]
            evaluated = stats["evaluated"]
            wrong = stats["wrong"]
            positions[f"position_{position}"] = {
                "evaluated": evaluated,
                "wrong": wrong,
                "wrong_rate": wrong / max(evaluated, 1),
                "share_of_wrong_positions": wrong / max(wrong_positions_total, 1),
            }

        return {
            "position_indexing": "left_to_right",
            "total_cases": total_cases,
            "wrong_cases": wrong_cases,
            "wrong_case_rate": wrong_cases / max(total_cases, 1),
            "parsed_wrong_cases": parsed_wrong_cases,
            "unparsed_cases": unparsed_cases,
            "sign_mismatch_cases": sign_mismatch_cases,
            "wrong_positions_total": wrong_positions_total,
            "positions": positions,
        }

    def _score_integer_cases(
        self,
        cases: list[tuple[str, int, int]],
    ) -> dict[str, object]:
        """Generate answers for integer prompts and compute exact-match accuracy."""
        generated_texts = self._generate_texts(
            [prompt for prompt, _, _ in cases],
            [max_tokens for _, _, max_tokens in cases],
        )
        correct = 0
        digit_error_counts = self._new_digit_error_counts()
        for (_, expected, _), generated in zip(cases, generated_texts):
            extracted = self._extract_number_from_generation(generated)
            self._accumulate_digit_error_counts(digit_error_counts, expected, extracted)
            if extracted is not None and int(extracted) == expected:
                correct += 1
        total = len(cases)
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / max(total, 1),
            "digit_error_profile": self._finalize_digit_error_profile(digit_error_counts),
        }

    def _log_digit_error_profile(self, label: str, profile: dict[str, object]) -> None:
        """Log a compact summary of wrong-digit positions."""
        positions = profile["positions"]
        if not positions:
            logger.info("  %s wrong-digit positions: none", label)
            return

        summary = ", ".join(
            f"{position}={stats['wrong_rate']:.1%}"
            for position, stats in positions.items()
        )
        logger.info(
            "  %s wrong-digit positions (%s): %s",
            label,
            profile["position_indexing"],
            summary,
        )

    def _add_pair_metric_summaries(
        self,
        results: dict[str, object],
        op_name: str,
        pair_results: dict[str, float],
    ) -> None:
        """Store per-pair metrics and aggregate summaries for one operation."""
        diagonal_values = []
        cross_values = []
        for pair_key, accuracy in pair_results.items():
            left_digits, right_digits = pair_key.split("x")
            if left_digits == right_digits:
                diagonal_values.append(accuracy)
            else:
                cross_values.append(accuracy)

        all_values = list(pair_results.values())
        results[f"{op_name}_digit_pairs"] = pair_results
        results[f"{op_name}_overall_mean"] = sum(all_values) / max(len(all_values), 1)
        results[f"{op_name}_diagonal_mean"] = sum(diagonal_values) / max(len(diagonal_values), 1)
        results[f"{op_name}_cross_digit_mean"] = sum(cross_values) / max(len(cross_values), 1)

    def exact_match_accuracy(
        self,
        num_samples: int | None = None,
        seed: int = 12345,
    ) -> dict[str, object]:
        """Test exact-match accuracy on ordered digit-pair arithmetic.

        Returns:
            Dict containing legacy same-digit keys plus full digit-pair metrics.
        """
        import random

        rng = random.Random(seed)
        self._prepare_model()

        samples = num_samples or self.config.num_samples_per_config
        results: dict[str, object] = {}

        for op in ["add", "sub", "mul"]:
            pair_results: dict[str, float] = {}
            op_digit_error_counts = self._new_digit_error_counts()
            for digits_a, digits_b in self._digit_pairs():
                cases: list[tuple[str, int, int]] = []

                for _ in range(samples):
                    a = _sample_number(digits_a, rng)
                    b = _sample_number(digits_b, rng)

                    # Signed subtraction is not yet supported by the injection format.
                    if op == "sub" and a < b:
                        a, b = b, a

                    expected = _compute_expected(op, a, b)
                    if expected is None or abs(expected) > 10**10:
                        continue

                    symbol = OP_SYMBOLS[op]
                    prompt = f"{a}{symbol}{b}="
                    cases.append((prompt, expected, len(str(abs(expected))) + 5))

                score = self._score_integer_cases(cases)
                pair_key = f"{digits_a}x{digits_b}"
                pair_results[pair_key] = score["accuracy"]
                logger.info(
                    f"  {op}_{pair_key}: {score['accuracy']:.1%} "
                    f"({score['correct']}/{score['total']})"
                )
                self._merge_digit_error_counts(
                    op_digit_error_counts,
                    {
                        "total_cases": score["digit_error_profile"]["total_cases"],
                        "wrong_cases": score["digit_error_profile"]["wrong_cases"],
                        "parsed_wrong_cases": score["digit_error_profile"]["parsed_wrong_cases"],
                        "unparsed_cases": score["digit_error_profile"]["unparsed_cases"],
                        "sign_mismatch_cases": score["digit_error_profile"]["sign_mismatch_cases"],
                        "positions": {
                            int(position.removeprefix("position_")): {
                                "evaluated": stats["evaluated"],
                                "wrong": stats["wrong"],
                            }
                            for position, stats in score["digit_error_profile"]["positions"].items()
                        },
                    },
                )

                if digits_a == digits_b:
                    legacy_key = f"{op}_{digits_a}digit"
                    results[legacy_key] = score["accuracy"]

            self._add_pair_metric_summaries(results, op, pair_results)
            results[f"{op}_wrong_digit_distribution"] = self._finalize_digit_error_profile(
                op_digit_error_counts
            )
            self._log_digit_error_profile(op, results[f"{op}_wrong_digit_distribution"])

        return results

    def exponentiation_accuracy(
        self,
        num_samples: int | None = None,
        seed: int = 12345,
    ) -> dict[str, float]:
        """Test exact-match accuracy on exponentiation."""
        import random

        rng = random.Random(seed)
        self._prepare_model()

        samples = num_samples or self.config.num_samples_per_config
        exp_cases: list[tuple[str, int, int]] = []
        for _ in range(samples):
            a = rng.randint(2, 20)
            b = rng.randint(0, 10)
            expected = _compute_expected("exp", a, b)
            if expected is None:
                continue
            prompt = f"{a}^{b}="
            exp_cases.append((prompt, expected, len(str(expected)) + 5))

        score = self._score_integer_cases(exp_cases)
        logger.info(f"  exp_accuracy: {score['accuracy']:.1%} ({score['correct']}/{score['total']})")
        return {
            "exp_accuracy": score["accuracy"],
            "exp_wrong_digit_distribution": score["digit_error_profile"],
        }

    def division_accuracy(
        self,
        num_samples: int | None = None,
        seed: int = 54321,
    ) -> dict[str, object]:
        """Test exact-match accuracy on exact division.

        Evaluates ordered (divisor_digits, quotient_digits) pairs.

        Returns:
            Dict with division accuracy metrics.
        """
        import random

        rng = random.Random(seed)
        self._prepare_model()

        samples = num_samples or self.config.num_samples_per_config
        results: dict[str, object] = {}
        pair_results: dict[str, float] = {}
        div_digit_error_counts = self._new_digit_error_counts()

        for divisor_digits, quotient_digits in self._digit_pairs():
            cases: list[tuple[str, int, int]] = []
            attempts = 0
            max_attempts = max(samples * 20, 20)

            while len(cases) < samples and attempts < max_attempts:
                attempts += 1
                b = _sample_number(divisor_digits, rng, min_value=1)
                quotient = _sample_number(quotient_digits, rng, min_value=1)
                a = b * quotient
                if a > 10**10:
                    continue

                prompt = f"{a}/{b}="
                cases.append((prompt, quotient, len(str(quotient)) + 5))

            score = self._score_integer_cases(cases)
            pair_key = f"{divisor_digits}x{quotient_digits}"
            pair_results[pair_key] = score["accuracy"]
            logger.info(
                f"  div_{pair_key}: {score['accuracy']:.1%} "
                f"({score['correct']}/{score['total']})"
            )
            self._merge_digit_error_counts(
                div_digit_error_counts,
                {
                    "total_cases": score["digit_error_profile"]["total_cases"],
                    "wrong_cases": score["digit_error_profile"]["wrong_cases"],
                    "parsed_wrong_cases": score["digit_error_profile"]["parsed_wrong_cases"],
                    "unparsed_cases": score["digit_error_profile"]["unparsed_cases"],
                    "sign_mismatch_cases": score["digit_error_profile"]["sign_mismatch_cases"],
                    "positions": {
                        int(position.removeprefix("position_")): {
                            "evaluated": stats["evaluated"],
                            "wrong": stats["wrong"],
                        }
                        for position, stats in score["digit_error_profile"]["positions"].items()
                    },
                },
            )

            if divisor_digits == quotient_digits:
                legacy_key = f"div_{divisor_digits}digit"
                results[legacy_key] = score["accuracy"]

        self._add_pair_metric_summaries(results, "div", pair_results)
        results["div_wrong_digit_distribution"] = self._finalize_digit_error_profile(
            div_digit_error_counts
        )
        self._log_digit_error_profile("div", results["div_wrong_digit_distribution"])

        return results

    def transcendental_accuracy(
        self,
        num_samples: int | None = None,
        tolerance: float = 1e-4,
        seed: int = 67890,
    ) -> dict[str, float]:
        """Test approximate accuracy on transcendental functions.

        Generates prompts like "sin(1.047) =" and checks if the model
        produces a result within tolerance of the correct value.

        Returns:
            Dict mapping function names to accuracy.
        """
        import random

        rng = random.Random(seed)
        self._prepare_model()

        samples = num_samples or self.config.num_samples_per_config
        results: dict[str, float] = {}

        # Define functions with their domains and evaluators
        functions = {
            "sin": (lambda: rng.uniform(0, 2 * math.pi), math.sin),
            "cos": (lambda: rng.uniform(0, 2 * math.pi), math.cos),
            "tan": (lambda: rng.uniform(-1.4, 1.4), math.tan),
            "exp": (lambda: rng.uniform(-5, 10), math.exp),
            "ln": (lambda: rng.uniform(0.1, 1000), math.log),
            "sqrt": (lambda: rng.uniform(0, 100000), math.sqrt),
        }

        for fname, (sampler, evaluator) in functions.items():
            cases: list[tuple[str, float]] = []

            for _ in range(samples):
                x = sampler()
                expected = evaluator(x)
                if abs(expected) > 1e9:
                    continue

                x_str = f"{x:.6f}"
                prompt = f"{fname}({x_str})="
                cases.append((prompt, expected))

            generated_texts = self._generate_texts(
                [prompt for prompt, _ in cases],
            )
            correct = 0
            for (_, expected), generated in zip(cases, generated_texts):
                extracted = self._extract_float_from_generation(generated)

                if extracted is not None:
                    if abs(expected) < 1e-6:
                        if abs(extracted) < tolerance:
                            correct += 1
                    elif abs(extracted - expected) / max(abs(expected), 1e-10) < tolerance:
                        correct += 1
            total = len(cases)

            accuracy = correct / max(total, 1)
            results[f"{fname}_accuracy"] = accuracy
            logger.info(f"  {fname}: {accuracy:.1%} ({correct}/{total})")

        return results

    def float_arithmetic_accuracy(
        self,
        num_samples: int | None = None,
        tolerance: float = 1e-4,
        seed: int = 11111,
    ) -> dict[str, float]:
        """Test approximate accuracy on floating-point arithmetic.

        Returns:
            Dict mapping operation names to accuracy.
        """
        import random

        rng = random.Random(seed)
        self._prepare_model()

        samples = num_samples or self.config.num_samples_per_config
        results: dict[str, float] = {}

        float_ops = {
            "float_add": ("+", lambda a, b: a + b),
            "float_sub": ("-", lambda a, b: a - b),
            "float_mul": ("*", lambda a, b: a * b),
            "float_div": ("/", lambda a, b: a / b if b != 0 else None),
        }

        for op_name, (symbol, compute) in float_ops.items():
            cases: list[tuple[str, float]] = []

            for _ in range(samples):
                a = round(rng.uniform(0.1, 1000), rng.randint(1, 4))
                b = round(rng.uniform(0.1, 1000), rng.randint(1, 4))
                expected = compute(a, b)
                if expected is None or abs(expected) > 1e9:
                    continue

                prompt = f"{a}{symbol}{b}="
                cases.append((prompt, expected))

            generated_texts = self._generate_texts(
                [prompt for prompt, _ in cases],
            )
            correct = 0
            for (_, expected), generated in zip(cases, generated_texts):
                extracted = self._extract_float_from_generation(generated)

                if extracted is not None:
                    if abs(expected) < 1e-6:
                        if abs(extracted) < tolerance:
                            correct += 1
                    elif abs(extracted - expected) / max(abs(expected), 1e-10) < tolerance:
                        correct += 1
            total = len(cases)

            accuracy = correct / max(total, 1)
            results[f"{op_name}_accuracy"] = accuracy
            logger.info(f"  {op_name}: {accuracy:.1%} ({correct}/{total})")

        return results

    def multi_step_accuracy(
        self,
        num_samples: int | None = None,
        seed: int = 22222,
    ) -> dict[str, float]:
        """Test accuracy on multi-step arithmetic expressions.

        Generates 2-step prompts like "(3 + 4) * 5 =" and checks
        if the final result is correct.

        Returns:
            Dict with multi-step accuracy metrics.
        """
        import random

        rng = random.Random(seed)
        self._prepare_model()

        samples = num_samples or self.config.num_samples_per_config
        ops = ["+", "-", "*"]

        cases: list[tuple[str, int, int]] = []

        for _ in range(samples):
            a = rng.randint(1, 99)
            b = rng.randint(1, 99)
            c = rng.randint(1, 99)
            op1 = rng.choice(ops)
            op2 = rng.choice(ops)

            intermediate = eval(f"{a} {op1} {b}")
            expected = eval(f"{intermediate} {op2} {c}")
            if abs(expected) > 10**10:
                continue

            prompt = f"({a}{op1}{b}){op2}{c}="
            cases.append((prompt, expected, len(str(abs(expected))) + 5))

        generated_texts = self._generate_texts(
            [prompt for prompt, _, _ in cases],
            [max_tokens for _, _, max_tokens in cases],
        )
        correct = 0
        for (_, expected, _), generated in zip(cases, generated_texts):
            extracted = self._extract_number_from_generation(generated)
            if extracted is not None and int(extracted) == expected:
                correct += 1
        total = len(cases)

        accuracy = correct / max(total, 1)
        logger.info(f"  multi_step_2: {accuracy:.1%} ({correct}/{total})")
        return {"multi_step_2_accuracy": accuracy}

    def _zero_injection(self) -> dict:
        """Zero all injection projection weights/biases and return saved state."""
        saved = {}
        for key, arb in self.model.arbs.items():
            proj = arb.inject.projection
            if isinstance(proj, nn.Sequential):
                # MLP injection: save and zero all layers
                layer_states = []
                for layer in proj:
                    if hasattr(layer, "weight"):
                        layer_states.append({
                            "weight": layer.weight.data.clone(),
                            "bias": layer.bias.data.clone(),
                        })
                        layer.weight.data.zero_()
                        layer.bias.data.zero_()
                saved[key] = {"type": "sequential", "layers": layer_states}
            else:
                saved[key] = {
                    "type": "linear",
                    "weight": proj.weight.data.clone(),
                    "bias": proj.bias.data.clone(),
                }
                proj.weight.data.zero_()
                proj.bias.data.zero_()
        return saved

    def _restore_injection(self, saved: dict) -> None:
        """Restore injection projection weights from saved state."""
        for key, arb in self.model.arbs.items():
            state = saved[key]
            proj = arb.inject.projection
            if state["type"] == "sequential":
                idx = 0
                for layer in proj:
                    if hasattr(layer, "weight"):
                        layer.weight.data.copy_(state["layers"][idx]["weight"])
                        layer.bias.data.copy_(state["layers"][idx]["bias"])
                        idx += 1
            else:
                proj.weight.data.copy_(state["weight"])
                proj.bias.data.copy_(state["bias"])

    def _zero_lora(self) -> dict | None:
        """Zero LoRA A matrix (disabling LoRA) and return saved state."""
        if not hasattr(self.model, "lora_head") or self.model.lora_head is None:
            return None
        saved = {
            "lora_A": self.model.lora_head.lora_A.data.clone(),
            "lora_B": self.model.lora_head.lora_B.data.clone(),
        }
        self.model.lora_head.lora_A.data.zero_()
        self.model.lora_head.lora_B.data.zero_()
        return saved

    def _restore_lora(self, saved: dict | None) -> None:
        """Restore LoRA weights from saved state."""
        if saved is None or self.model.lora_head is None:
            return
        self.model.lora_head.lora_A.data.copy_(saved["lora_A"])
        self.model.lora_head.lora_B.data.copy_(saved["lora_B"])

    def _zero_layer_lora(self) -> dict | None:
        """Zero all layer LoRA parameters and return saved state."""
        if not hasattr(self.model, "lora_layers") or self.model.lora_layers is None:
            return None
        saved = {}
        for key, lora_module in self.model.lora_layers.items():
            saved[key] = {
                "lora_A": lora_module.lora_A.data.clone(),
                "lora_B": lora_module.lora_B.data.clone(),
            }
            lora_module.lora_A.data.zero_()
            lora_module.lora_B.data.zero_()
        return saved

    def _restore_layer_lora(self, saved: dict | None) -> None:
        """Restore layer LoRA weights from saved state."""
        if saved is None or self.model.lora_layers is None:
            return
        for key, lora_module in self.model.lora_layers.items():
            lora_module.lora_A.data.copy_(saved[key]["lora_A"])
            lora_module.lora_B.data.copy_(saved[key]["lora_B"])

    def _run_accuracy_test(self, num_samples: int, rng) -> int:
        """Run a quick accuracy test on 2-digit addition, return correct count."""
        cases: list[tuple[str, int]] = []
        for _ in range(num_samples):
            a, b = _sample_operands(2, rng)
            prompt = f"{a}+{b}="
            expected = a + b
            cases.append((prompt, expected))

        generated_texts = self._generate_texts(
            [prompt for prompt, _ in cases],
        )
        correct = 0
        for (_, expected), generated in zip(cases, generated_texts):
            extracted = self._extract_number_from_generation(generated)
            if extracted is not None and int(extracted) == expected:
                correct += 1
        return correct

    def ablation_test(self, num_samples: int = 50, seed: int = 99) -> dict[str, float]:
        """Zero out injection, confirm accuracy reverts to baseline.

        Handles both linear and MLP injection projections.

        Returns:
            Dict with 'ablated' and 'normal' accuracy for comparison.
        """
        import random

        rng = random.Random(seed)
        self._prepare_model()

        normal_correct = self._run_accuracy_test(num_samples, rng)

        saved = self._zero_injection()
        rng = random.Random(seed)
        ablated_correct = self._run_accuracy_test(num_samples, rng)
        self._restore_injection(saved)

        normal_acc = normal_correct / num_samples
        ablated_acc = ablated_correct / num_samples
        logger.info(f"Ablation: normal={normal_acc:.1%}, ablated={ablated_acc:.1%}")
        return {"normal_accuracy": normal_acc, "ablated_accuracy": ablated_acc}

    def four_way_ablation(
        self,
        num_samples: int = 50,
        seed: int = 99,
    ) -> dict[str, float]:
        """Four-way ablation proving the ARB is the source of improvement.

        Conditions:
        1. Full (ARB + LoRA): expected ~100% accuracy
        2. LoRA only (ARB zeroed): LoRA can't do arithmetic alone
        3. ARB only (LoRA zeroed): some improvement from injection
        4. Baseline (both zeroed): base model accuracy

        Returns:
            Dict mapping condition names to accuracy.
        """
        import random

        self._prepare_model()

        results = {}

        # 1. Full model (ARB + LoRA)
        rng = random.Random(seed)
        full_correct = self._run_accuracy_test(num_samples, rng)
        results["full"] = full_correct / num_samples
        logger.info(f"  Full (ARB+LoRA): {results['full']:.1%}")

        # 2. LoRA only (zero ARB injection)
        saved_arb = self._zero_injection()
        rng = random.Random(seed)
        lora_only_correct = self._run_accuracy_test(num_samples, rng)
        results["lora_only"] = lora_only_correct / num_samples
        logger.info(f"  LoRA only: {results['lora_only']:.1%}")
        self._restore_injection(saved_arb)

        # 3. ARB only (zero head LoRA + layer LoRA)
        saved_lora = self._zero_lora()
        saved_layer_lora = self._zero_layer_lora()
        rng = random.Random(seed)
        arb_only_correct = self._run_accuracy_test(num_samples, rng)
        results["arb_only"] = arb_only_correct / num_samples
        logger.info(f"  ARB only: {results['arb_only']:.1%}")
        self._restore_layer_lora(saved_layer_lora)
        self._restore_lora(saved_lora)

        # 4. Baseline (zero all: ARB injection + head LoRA + layer LoRA)
        saved_arb = self._zero_injection()
        saved_lora = self._zero_lora()
        saved_layer_lora = self._zero_layer_lora()
        rng = random.Random(seed)
        baseline_correct = self._run_accuracy_test(num_samples, rng)
        results["baseline"] = baseline_correct / num_samples
        logger.info(f"  Baseline: {results['baseline']:.1%}")
        self._restore_layer_lora(saved_layer_lora)
        self._restore_lora(saved_lora)
        self._restore_injection(saved_arb)

        return results

    @torch.no_grad()
    def perplexity_test(self, eval_texts: list[str], max_samples: int = 200) -> dict[str, float]:
        """Compute perplexity on evaluation texts.

        Since the base model is frozen, perplexity should be nearly identical
        to the unmodified GPT-2.

        Args:
            eval_texts: List of text samples for perplexity evaluation
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dict with 'perplexity' value
        """
        self._prepare_model()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        total_loss = 0.0
        total_tokens = 0

        texts = eval_texts[:max_samples]
        loader = DataLoader(texts, batch_size=self.batch_size, shuffle=False)

        for batch_texts in tqdm(loader, desc="Perplexity", disable=len(texts) < 10):
            encoding = self.tokenizer(
                list(batch_texts),
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            valid_rows = attention_mask.sum(dim=1) >= 2
            if not valid_rows.any():
                continue

            input_ids = input_ids[valid_rows].to(self.device)
            attention_mask = attention_mask[valid_rows].to(self.device)
            labels = input_ids.masked_fill(attention_mask == 0, -100)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
            num_tokens = labels[:, 1:].ne(-100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        logger.info(f"Perplexity: {perplexity:.2f} (avg loss: {avg_loss:.4f})")
        return {"perplexity": perplexity, "avg_loss": avg_loss}

    def full_evaluation(
        self,
        eval_texts: list[str] | None = None,
    ) -> dict[str, object]:
        """Run the currently enabled integer evaluation suite."""
        results: dict[str, object] = {}

        logger.info("=== Exact Match Accuracy ===")
        results["exact_match"] = self.exact_match_accuracy()

        logger.info("=== Division Accuracy ===")
        results["division"] = self.division_accuracy()

        return results
