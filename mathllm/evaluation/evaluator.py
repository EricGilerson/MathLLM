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

    def _resolve_num_samples(self, num_samples: int | None = None) -> int:
        """Resolve the sample count for an evaluation run."""
        return int(num_samples or self.config.num_samples_per_config)

    @staticmethod
    def _build_integer_case(prompt: str, expected: int) -> dict[str, object]:
        """Build a single integer evaluation case."""
        answer_text = str(expected)
        return {
            "prompt": prompt,
            "expected": expected,
            "answer_text": answer_text,
            "max_new_tokens": len(str(abs(expected))) + 5,
        }

    def _build_exact_match_cells(
        self,
        num_samples: int | None = None,
        seed: int = 12345,
    ) -> list[dict[str, object]]:
        """Build integer arithmetic cells for add/sub/mul evaluation."""
        import random

        rng = random.Random(seed)
        samples = self._resolve_num_samples(num_samples)
        cells: list[dict[str, object]] = []

        for op in ["add", "sub", "mul"]:
            symbol = OP_SYMBOLS[op]
            for digits_a, digits_b in self._digit_pairs():
                cases: list[dict[str, object]] = []

                for _ in range(samples):
                    a = _sample_number(digits_a, rng)
                    b = _sample_number(digits_b, rng)

                    # Signed subtraction is not yet supported by the injection format.
                    if op == "sub" and a < b:
                        a, b = b, a

                    expected = _compute_expected(op, a, b)
                    if expected is None or abs(expected) > 10**10:
                        continue

                    cases.append(self._build_integer_case(f"{a}{symbol}{b}=", expected))

                pair_key = f"{digits_a}x{digits_b}"
                cells.append(
                    {
                        "operation": op,
                        "digit_pair": pair_key,
                        "legacy_key": f"{op}_{digits_a}digit" if digits_a == digits_b else None,
                        "cases": cases,
                    }
                )

        return cells

    def _build_division_cells(
        self,
        num_samples: int | None = None,
        seed: int = 54321,
    ) -> list[dict[str, object]]:
        """Build exact-division evaluation cells."""
        import random

        rng = random.Random(seed)
        samples = self._resolve_num_samples(num_samples)
        cells: list[dict[str, object]] = []

        for divisor_digits, quotient_digits in self._digit_pairs():
            cases: list[dict[str, object]] = []
            attempts = 0
            max_attempts = max(samples * 20, 20)

            while len(cases) < samples and attempts < max_attempts:
                attempts += 1
                b = _sample_number(divisor_digits, rng, min_value=1)
                quotient = _sample_number(quotient_digits, rng, min_value=1)
                a = b * quotient
                if a > 10**10:
                    continue

                cases.append(self._build_integer_case(f"{a}/{b}=", quotient))

            pair_key = f"{divisor_digits}x{quotient_digits}"
            cells.append(
                {
                    "operation": "div",
                    "digit_pair": pair_key,
                    "legacy_key": (
                        f"div_{divisor_digits}digit"
                        if divisor_digits == quotient_digits else None
                    ),
                    "cases": cases,
                }
            )

        return cells

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

    def _profile_to_digit_error_counts(
        self,
        profile: dict[str, object],
    ) -> dict[str, object]:
        """Convert a finalized profile back into mergeable counters."""
        return {
            "total_cases": profile["total_cases"],
            "wrong_cases": profile["wrong_cases"],
            "parsed_wrong_cases": profile["parsed_wrong_cases"],
            "unparsed_cases": profile["unparsed_cases"],
            "sign_mismatch_cases": profile["sign_mismatch_cases"],
            "positions": {
                int(position.removeprefix("position_")): {
                    "evaluated": stats["evaluated"],
                    "wrong": stats["wrong"],
                }
                for position, stats in profile["positions"].items()
            },
        }

    def _evaluate_integer_cases(
        self,
        cases: list[dict[str, object]],
        include_examples: bool = False,
    ) -> dict[str, object]:
        """Generate answers for integer prompts and compute exact-match accuracy."""
        generated_texts = self._generate_texts(
            [str(case["prompt"]) for case in cases],
            [int(case["max_new_tokens"]) for case in cases],
        )
        correct = 0
        digit_error_counts = self._new_digit_error_counts()
        example_results: list[dict[str, object]] = []
        for case, generated in zip(cases, generated_texts):
            expected = int(case["expected"])
            extracted = self._extract_number_from_generation(generated)
            self._accumulate_digit_error_counts(digit_error_counts, expected, extracted)
            is_correct = extracted is not None and int(extracted) == expected
            if is_correct:
                correct += 1
            if include_examples:
                example_results.append(
                    {
                        "prompt": case["prompt"],
                        "expected_answer": case["answer_text"],
                        "arb_generation_text": generated,
                        "arb_extracted_answer": extracted,
                        "arb_correct": is_correct,
                    }
                )
        total = len(cases)
        result = {
            "correct": correct,
            "total": total,
            "accuracy": correct / max(total, 1),
            "digit_error_profile": self._finalize_digit_error_profile(digit_error_counts),
        }
        if include_examples:
            result["examples"] = example_results
        return result

    def _score_integer_cases(
        self,
        cases: list[dict[str, object]] | list[tuple[str, int, int]],
    ) -> dict[str, object]:
        """Backward-compatible aggregate integer scoring helper."""
        if cases and isinstance(cases[0], tuple):
            cases = [
                self._build_integer_case(prompt, expected)
                for prompt, expected, _ in cases
            ]
        return self._evaluate_integer_cases(cases, include_examples=False)

    def _evaluate_integer_cells(
        self,
        cells: list[dict[str, object]],
        include_examples: bool = False,
    ) -> list[dict[str, object]]:
        """Evaluate a list of integer cells and attach cell metadata."""
        self._prepare_model()

        cell_results: list[dict[str, object]] = []
        for cell in cells:
            cases = cell["cases"]
            score = self._evaluate_integer_cases(cases, include_examples=include_examples)
            logger.info(
                "  %s_%s: %.1f%% (%d/%d)",
                cell["operation"],
                cell["digit_pair"],
                score["accuracy"] * 100.0,
                score["correct"],
                score["total"],
            )

            cell_result = {
                "operation": cell["operation"],
                "digit_pair": cell["digit_pair"],
                "legacy_key": cell["legacy_key"],
                "correct": score["correct"],
                "total": score["total"],
                "accuracy": score["accuracy"],
                "digit_error_profile": score["digit_error_profile"],
            }
            if include_examples:
                examples = []
                for example in score["examples"]:
                    examples.append(
                        {
                            **example,
                            "operation": cell["operation"],
                            "digit_pair": cell["digit_pair"],
                            "cell_id": f"{cell['operation']}_{cell['digit_pair']}",
                        }
                    )
                cell_result["examples"] = examples
            cell_results.append(cell_result)

        return cell_results

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

    def _build_operation_results(
        self,
        op_name: str,
        cell_results: list[dict[str, object]],
    ) -> dict[str, object]:
        """Build the public metric structure for one integer operation."""
        results: dict[str, object] = {}
        pair_results: dict[str, float] = {}
        op_digit_error_counts = self._new_digit_error_counts()

        for cell in cell_results:
            pair_results[cell["digit_pair"]] = cell["accuracy"]
            self._merge_digit_error_counts(
                op_digit_error_counts,
                self._profile_to_digit_error_counts(cell["digit_error_profile"]),
            )
            if cell["legacy_key"] is not None:
                results[str(cell["legacy_key"])] = cell["accuracy"]

        self._add_pair_metric_summaries(results, op_name, pair_results)
        results[f"{op_name}_wrong_digit_distribution"] = self._finalize_digit_error_profile(
            op_digit_error_counts
        )
        self._log_digit_error_profile(op_name, results[f"{op_name}_wrong_digit_distribution"])
        return results

    def exact_match_accuracy(
        self,
        num_samples: int | None = None,
        seed: int = 12345,
    ) -> dict[str, object]:
        """Test exact-match accuracy on ordered digit-pair arithmetic.

        Returns:
            Dict containing legacy same-digit keys plus full digit-pair metrics.
        """
        cells = self._build_exact_match_cells(num_samples=num_samples, seed=seed)
        cell_results = self._evaluate_integer_cells(cells)

        results: dict[str, object] = {}
        for op in ["add", "sub", "mul"]:
            op_cell_results = [cell for cell in cell_results if cell["operation"] == op]
            results.update(self._build_operation_results(op, op_cell_results))
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

        samples = self._resolve_num_samples(num_samples)
        exp_cases: list[dict[str, object]] = []
        for _ in range(samples):
            a = rng.randint(2, 20)
            b = rng.randint(0, 10)
            expected = _compute_expected("exp", a, b)
            if expected is None:
                continue
            exp_cases.append(self._build_integer_case(f"{a}^{b}=", expected))

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
        cells = self._build_division_cells(num_samples=num_samples, seed=seed)
        cell_results = self._evaluate_integer_cells(cells)
        return self._build_operation_results("div", cell_results)

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

        samples = self._resolve_num_samples(num_samples)
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

        samples = self._resolve_num_samples(num_samples)
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

        samples = self._resolve_num_samples(num_samples)
        ops = ["+", "-", "*"]

        cases: list[dict[str, object]] = []

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

            cases.append(self._build_integer_case(f"({a}{op1}{b}){op2}{c}=", expected))

        generated_texts = self._generate_texts(
            [str(case["prompt"]) for case in cases],
            [int(case["max_new_tokens"]) for case in cases],
        )
        correct = 0
        for case, generated in zip(cases, generated_texts):
            expected = int(case["expected"])
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
        cases: list[dict[str, object]] = []
        for _ in range(num_samples):
            a, b = _sample_operands(2, rng)
            cases.append(self._build_integer_case(f"{a}+{b}=", a + b))

        generated_texts = self._generate_texts(
            [str(case["prompt"]) for case in cases],
        )
        correct = 0
        for case, generated in zip(cases, generated_texts):
            expected = int(case["expected"])
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

    @staticmethod
    def _extract_model_logits(outputs) -> torch.Tensor:
        """Extract logits from either a dict-like or HF model output."""
        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs.logits

    @staticmethod
    def _mean(values: list[float]) -> float:
        """Compute a mean with an empty-list guard."""
        return sum(values) / max(len(values), 1)

    @staticmethod
    def _median(values: list[float]) -> float | None:
        """Compute the median of a list."""
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    @staticmethod
    def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
        """Compute Pearson correlation."""
        if len(xs) != len(ys) or len(xs) < 2:
            return None
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        centered_x = [x - mean_x for x in xs]
        centered_y = [y - mean_y for y in ys]
        numerator = sum(x * y for x, y in zip(centered_x, centered_y))
        denom_x = math.sqrt(sum(x * x for x in centered_x))
        denom_y = math.sqrt(sum(y * y for y in centered_y))
        if denom_x == 0.0 or denom_y == 0.0:
            return None
        return numerator / (denom_x * denom_y)

    @staticmethod
    def _rank_with_average_ties(values: list[float]) -> list[float]:
        """Assign 1-based average ranks with tie handling."""
        sorted_pairs = sorted(enumerate(values), key=lambda item: item[1])
        ranks = [0.0] * len(values)
        idx = 0
        while idx < len(sorted_pairs):
            end = idx + 1
            while end < len(sorted_pairs) and sorted_pairs[end][1] == sorted_pairs[idx][1]:
                end += 1
            avg_rank = (idx + 1 + end) / 2.0
            for tied_idx in range(idx, end):
                original_index = sorted_pairs[tied_idx][0]
                ranks[original_index] = avg_rank
            idx = end
        return ranks

    def _spearman_correlation(self, xs: list[float], ys: list[float]) -> float | None:
        """Compute Spearman correlation via average ranks."""
        if len(xs) != len(ys) or len(xs) < 2:
            return None
        return self._pearson_correlation(
            self._rank_with_average_ties(xs),
            self._rank_with_average_ties(ys),
        )

    @staticmethod
    def _detect_tokenization_mode(tokenizer) -> str:
        """Report whether arithmetic numbers tokenize per digit or as single tokens."""
        if not hasattr(tokenizer, "encode"):
            return "unknown"
        token_ids = tokenizer.encode("300", add_special_tokens=False)
        if len(token_ids) > 1:
            return "per_digit"
        return "single_token_number"

    @torch.no_grad()
    def _score_base_model_answer_logprobs(
        self,
        examples: list[dict[str, object]],
        base_model,
        base_tokenizer,
    ) -> list[dict[str, object]]:
        """Teacher-force the frozen base model and score answer-token log-probs."""
        if not examples:
            return []

        base_model.eval()
        base_model.to(self.device)

        pad_token_id = getattr(base_tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(base_tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Base tokenizer must define pad_token_id or eos_token_id")

        prepared: list[dict[str, object]] = []
        for example in examples:
            prompt = str(example["prompt"])
            answer_text = str(example["expected_answer"])
            full_text = prompt + answer_text

            prompt_ids = base_tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = base_tokenizer.encode(answer_text, add_special_tokens=False)
            full_ids = base_tokenizer.encode(full_text, add_special_tokens=False)

            prompt_len = len(prompt_ids)
            answer_len = len(answer_ids)
            if answer_len == 0:
                raise ValueError(f"Expected non-empty answer tokenization for {full_text!r}")
            if full_ids[:prompt_len] != prompt_ids or full_ids[prompt_len:] != answer_ids:
                raise ValueError(
                    "Prior-logit analysis requires prompt+answer tokenization to split "
                    "cleanly into prompt and answer tokens."
                )

            prepared.append(
                {
                    **example,
                    "prompt_len": prompt_len,
                    "answer_len": answer_len,
                    "input_ids": full_ids,
                    "answer_ids": answer_ids,
                }
            )

        scored_examples: list[dict[str, object]] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start:start + self.batch_size]
            max_len = max(len(record["input_ids"]) for record in batch)
            input_ids = torch.full(
                (len(batch), max_len),
                pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros(
                (len(batch), max_len),
                dtype=torch.long,
                device=self.device,
            )

            for row, record in enumerate(batch):
                record_input_ids = torch.tensor(
                    record["input_ids"],
                    dtype=torch.long,
                    device=self.device,
                )
                seq_len = int(record_input_ids.numel())
                input_ids[row, :seq_len] = record_input_ids
                attention_mask[row, :seq_len] = 1

            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = self._extract_model_logits(outputs)
            shift_log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)

            for row, record in enumerate(batch):
                prompt_len = int(record["prompt_len"])
                answer_len = int(record["answer_len"])
                answer_start = prompt_len - 1
                answer_end = answer_start + answer_len
                answer_log_probs = shift_log_probs[row, answer_start:answer_end]
                answer_ids = torch.tensor(
                    record["answer_ids"],
                    dtype=torch.long,
                    device=self.device,
                )

                correct_log_probs = answer_log_probs.gather(
                    dim=1,
                    index=answer_ids.unsqueeze(1),
                ).squeeze(1)
                top1_log_probs, top1_token_ids = answer_log_probs.max(dim=1)
                top1_wrong_fraction = (
                    (top1_token_ids != answer_ids).float().mean().item()
                )

                scored_examples.append(
                    {
                        **{
                            key: value for key, value in record.items()
                            if key not in {"prompt_len", "answer_len", "input_ids", "answer_ids"}
                        },
                        "num_answer_tokens": answer_len,
                        "correct_logprob_mean": correct_log_probs.mean().item(),
                        "correct_prob_mean": correct_log_probs.exp().mean().item(),
                        "top1_logprob_mean": top1_log_probs.mean().item(),
                        "logprob_gap_mean": (top1_log_probs - correct_log_probs).mean().item(),
                        "top1_wrong_fraction": top1_wrong_fraction,
                    }
                )

        return scored_examples

    def _summarize_example_group(
        self,
        examples: list[dict[str, object]],
    ) -> dict[str, object]:
        """Summarize a slice of per-example prior scores."""
        correct_logprobs = [float(example["correct_logprob_mean"]) for example in examples]
        top1_logprobs = [float(example["top1_logprob_mean"]) for example in examples]
        logprob_gaps = [float(example["logprob_gap_mean"]) for example in examples]
        top1_wrong = [float(example["top1_wrong_fraction"]) for example in examples]
        return {
            "num_examples": len(examples),
            "mean_correct_logprob": self._mean(correct_logprobs),
            "median_correct_logprob": self._median(correct_logprobs),
            "mean_top1_logprob": self._mean(top1_logprobs),
            "median_top1_logprob": self._median(top1_logprobs),
            "mean_logprob_gap": self._mean(logprob_gaps),
            "mean_top1_wrong_fraction": self._mean(top1_wrong),
        }

    def competing_prior_logit_analysis(
        self,
        cell_results: list[dict[str, object]],
        base_model,
        base_tokenizer,
    ) -> dict[str, object]:
        """Measure frozen-base answer confidence and correlate it with ARB failures."""
        logger.info("=== Competing-Prior Logit Analysis ===")

        all_examples = [
            example
            for cell in cell_results
            for example in cell.get("examples", [])
        ]
        scored_examples = self._score_base_model_answer_logprobs(
            all_examples,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
        )

        per_cell: list[dict[str, object]] = []
        cell_to_examples: dict[str, list[dict[str, object]]] = defaultdict(list)
        for example in scored_examples:
            cell_to_examples[str(example["cell_id"])].append(example)

        for cell in cell_results:
            cell_id = f"{cell['operation']}_{cell['digit_pair']}"
            examples = cell_to_examples[cell_id]
            correct_logprobs = [float(example["correct_logprob_mean"]) for example in examples]
            correct_probs = [float(example["correct_prob_mean"]) for example in examples]
            top1_logprobs = [float(example["top1_logprob_mean"]) for example in examples]
            logprob_gaps = [float(example["logprob_gap_mean"]) for example in examples]
            top1_wrong = [float(example["top1_wrong_fraction"]) for example in examples]
            num_answer_tokens = sum(int(example["num_answer_tokens"]) for example in examples)
            arb_error_count = int(cell["total"]) - int(cell["correct"])

            per_cell.append(
                {
                    "cell_id": cell_id,
                    "operation": cell["operation"],
                    "digit_pair": cell["digit_pair"],
                    "num_examples": len(examples),
                    "num_answer_tokens": num_answer_tokens,
                    "base_mean_correct_logprob": self._mean(correct_logprobs),
                    "base_mean_correct_prob": self._mean(correct_probs),
                    "base_mean_top1_logprob": self._mean(top1_logprobs),
                    "base_mean_logprob_gap": self._mean(logprob_gaps),
                    "base_mean_top1_wrong_fraction": self._mean(top1_wrong),
                    "arb_accuracy": float(cell["accuracy"]),
                    "arb_error_rate": arb_error_count / max(int(cell["total"]), 1),
                    "arb_error_count": arb_error_count,
                }
            )

        per_cell.sort(key=lambda item: (str(item["operation"]), str(item["digit_pair"])))
        xs = [float(cell["base_mean_correct_logprob"]) for cell in per_cell]
        ys = [float(cell["arb_error_rate"]) for cell in per_cell]

        arb_failures = [example for example in scored_examples if not bool(example["arb_correct"])]
        arb_successes = [example for example in scored_examples if bool(example["arb_correct"])]

        logger.info(
            "  prior/cell correlation: pearson=%s spearman=%s",
            (
                f"{self._pearson_correlation(xs, ys):.4f}"
                if self._pearson_correlation(xs, ys) is not None else "n/a"
            ),
            (
                f"{self._spearman_correlation(xs, ys):.4f}"
                if self._spearman_correlation(xs, ys) is not None else "n/a"
            ),
        )

        return {
            "metric": {
                "primary": (
                    "Mean frozen-base log-probability assigned to the correct "
                    "answer tokens after '='."
                ),
                "secondary": {
                    "top1_logprob_mean": (
                        "Mean frozen-base log-probability of the model's top-1 "
                        "predicted answer token."
                    ),
                    "logprob_gap_mean": "Mean top1_logprob - correct_logprob.",
                    "top1_wrong_fraction": (
                        "Fraction of answer-token positions where the frozen base "
                        "model's top-1 token is not the correct answer token."
                    ),
                },
            },
            "tokenization_mode": self._detect_tokenization_mode(base_tokenizer),
            "num_cells": len(per_cell),
            "num_examples": len(scored_examples),
            "per_cell": per_cell,
            "error_vs_success_summary": {
                "arb_failures": self._summarize_example_group(arb_failures),
                "arb_successes": self._summarize_example_group(arb_successes),
            },
            "correlation": {
                "x_metric": "base_mean_correct_logprob",
                "y_metric": "arb_error_rate",
                "pearson_r": self._pearson_correlation(xs, ys),
                "spearman_r": self._spearman_correlation(xs, ys),
            },
        }

    def full_evaluation(
        self,
        eval_texts: list[str] | None = None,
        include_prior_logit_analysis: bool = False,
        base_model=None,
        base_tokenizer=None,
    ) -> dict[str, object]:
        """Run the currently enabled integer evaluation suite."""
        results: dict[str, object] = {}

        if include_prior_logit_analysis and (base_model is None or base_tokenizer is None):
            raise ValueError(
                "Prior-logit analysis requires both a frozen base model and tokenizer."
            )

        if include_prior_logit_analysis:
            logger.info("=== Exact Match Accuracy ===")
            exact_match_cells = self._evaluate_integer_cells(
                self._build_exact_match_cells(),
                include_examples=True,
            )
            results["exact_match"] = {}
            for op in ["add", "sub", "mul"]:
                op_cell_results = [cell for cell in exact_match_cells if cell["operation"] == op]
                results["exact_match"].update(self._build_operation_results(op, op_cell_results))

            logger.info("=== Division Accuracy ===")
            division_cells = self._evaluate_integer_cells(
                self._build_division_cells(),
                include_examples=True,
            )
            results["division"] = self._build_operation_results("div", division_cells)
            results["competing_prior_logit_analysis"] = self.competing_prior_logit_analysis(
                exact_match_cells + division_cells,
                base_model=base_model,
                base_tokenizer=base_tokenizer,
            )
            return results

        logger.info("=== Exact Match Accuracy ===")
        results["exact_match"] = self.exact_match_accuracy()

        logger.info("=== Division Accuracy ===")
        results["division"] = self.division_accuracy()

        return results
