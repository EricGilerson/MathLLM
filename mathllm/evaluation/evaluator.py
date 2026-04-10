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

OP_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "exp": "**", "div": "/"}


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

    def _generate_text(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Generate text from a prompt."""
        max_tokens = max_new_tokens or self.config.max_new_tokens
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, max_new_tokens=max_tokens, greedy=True
            )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Return only the generated part (after prompt)
        return full_text[len(prompt):]

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

    def exact_match_accuracy(
        self,
        num_samples: int | None = None,
        seed: int = 12345,
    ) -> dict[str, float]:
        """Test exact-match accuracy on N-digit operations.

        For each digit count in [1, max_digits] and each operation,
        generates prompts like "347 + 291 =" and checks if the model
        produces the correct result.

        Returns:
            Dict mapping "{op}_{n}digit" to accuracy (0.0 to 1.0)
        """
        import random

        rng = random.Random(seed)
        self.model.eval()
        self.model.to(self.device)

        samples = num_samples or self.config.num_samples_per_config
        min_d, max_d = self.config.max_digits_range
        results: dict[str, float] = {}

        for n in range(min_d, max_d + 1):
            for op in ["add", "sub", "mul"]:
                correct = 0
                total = 0

                for _ in range(samples):
                    a, b = _sample_operands(n, rng)

                    # For multiplication, use smaller operands to stay in range
                    if op == "mul" and n > 5:
                        a, b = _sample_operands(min(n, 4), rng)

                    expected = _compute_expected(op, a, b)
                    if expected is None or abs(expected) > 10**10:
                        continue

                    symbol = OP_SYMBOLS[op]
                    prompt = f"{a} {symbol} {b} ="
                    generated = self._generate_text(
                        prompt,
                        max_new_tokens=len(str(abs(expected))) + 5,
                    )

                    extracted = self._extract_number_from_generation(generated)
                    if extracted is not None and int(extracted) == expected:
                        correct += 1
                    total += 1

                accuracy = correct / max(total, 1)
                key = f"{op}_{n}digit"
                results[key] = accuracy
                logger.info(f"  {key}: {accuracy:.1%} ({correct}/{total})")

        # Exponentiation (separate because operand sampling differs)
        for _ in range(samples):
            a = rng.randint(2, 20)
            b = rng.randint(0, 10)
            expected = _compute_expected("exp", a, b)
            if expected is None:
                continue
            prompt = f"{a} ** {b} ="
            generated = self._generate_text(
                prompt, max_new_tokens=len(str(expected)) + 5
            )
            extracted = self._extract_number_from_generation(generated)
            if extracted is not None and int(extracted) == expected:
                results.setdefault("exp_correct", 0)
                results["exp_correct"] = results.get("exp_correct", 0) + 1
            results.setdefault("exp_total", 0)
            results["exp_total"] = results.get("exp_total", 0) + 1

        if "exp_total" in results and results["exp_total"] > 0:
            results["exp_accuracy"] = results.pop("exp_correct", 0) / results.pop("exp_total")

        return results

    def division_accuracy(
        self,
        num_samples: int | None = None,
        seed: int = 54321,
    ) -> dict[str, float]:
        """Test exact-match accuracy on exact division.

        Generates prompts like "120 / 4 =" where divisor evenly divides dividend.

        Returns:
            Dict with division accuracy metrics.
        """
        import random

        rng = random.Random(seed)
        self.model.eval()
        self.model.to(self.device)

        samples = num_samples or self.config.num_samples_per_config
        min_d, max_d = self.config.max_digits_range
        results: dict[str, float] = {}

        for n in range(min_d, min(max_d + 1, 6)):  # up to 5-digit dividends
            correct = 0
            total = 0

            for _ in range(samples):
                b = rng.randint(2, 10 ** min(n, 3) - 1)
                quotient = rng.randint(1, max(1, (10**n - 1) // max(b, 1)))
                a = b * quotient
                if a > 10**10:
                    continue

                prompt = f"{a} / {b} ="
                generated = self._generate_text(
                    prompt, max_new_tokens=len(str(quotient)) + 5
                )
                extracted = self._extract_number_from_generation(generated)
                if extracted is not None and int(extracted) == quotient:
                    correct += 1
                total += 1

            accuracy = correct / max(total, 1)
            key = f"div_{n}digit"
            results[key] = accuracy
            logger.info(f"  {key}: {accuracy:.1%} ({correct}/{total})")

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
        self.model.eval()
        self.model.to(self.device)

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
            correct = 0
            total = 0

            for _ in range(samples):
                x = sampler()
                expected = evaluator(x)
                if abs(expected) > 1e9:
                    continue

                x_str = f"{x:.6f}"
                prompt = f"{fname}({x_str}) ="
                generated = self._generate_text(prompt, max_new_tokens=20)
                extracted = self._extract_float_from_generation(generated)

                if extracted is not None:
                    if abs(expected) < 1e-6:
                        if abs(extracted) < tolerance:
                            correct += 1
                    elif abs(extracted - expected) / max(abs(expected), 1e-10) < tolerance:
                        correct += 1
                total += 1

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
        self.model.eval()
        self.model.to(self.device)

        samples = num_samples or self.config.num_samples_per_config
        results: dict[str, float] = {}

        float_ops = {
            "float_add": ("+", lambda a, b: a + b),
            "float_sub": ("-", lambda a, b: a - b),
            "float_mul": ("*", lambda a, b: a * b),
            "float_div": ("/", lambda a, b: a / b if b != 0 else None),
        }

        for op_name, (symbol, compute) in float_ops.items():
            correct = 0
            total = 0

            for _ in range(samples):
                a = round(rng.uniform(0.1, 1000), rng.randint(1, 4))
                b = round(rng.uniform(0.1, 1000), rng.randint(1, 4))
                expected = compute(a, b)
                if expected is None or abs(expected) > 1e9:
                    continue

                prompt = f"{a} {symbol} {b} ="
                generated = self._generate_text(prompt, max_new_tokens=20)
                extracted = self._extract_float_from_generation(generated)

                if extracted is not None:
                    if abs(expected) < 1e-6:
                        if abs(extracted) < tolerance:
                            correct += 1
                    elif abs(extracted - expected) / max(abs(expected), 1e-10) < tolerance:
                        correct += 1
                total += 1

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
        self.model.eval()
        self.model.to(self.device)

        samples = num_samples or self.config.num_samples_per_config
        ops = ["+", "-", "*"]

        correct = 0
        total = 0

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

            prompt = f"({a} {op1} {b}) {op2} {c} ="
            generated = self._generate_text(
                prompt, max_new_tokens=len(str(abs(expected))) + 5
            )
            extracted = self._extract_number_from_generation(generated)
            if extracted is not None and int(extracted) == expected:
                correct += 1
            total += 1

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

    def _run_accuracy_test(
        self, num_samples: int, rng, pure_arithmetic: bool = False
    ) -> int:
        """Run a quick accuracy test on 2-digit addition, return correct count."""
        correct = 0
        for _ in range(num_samples):
            a, b = _sample_operands(2, rng)
            if pure_arithmetic:
                prompt = f"{a}+{b}="
            else:
                prompt = f"{a} + {b} ="
            expected = a + b
            generated = self._generate_text(prompt, max_new_tokens=10)
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
        self.model.eval()
        self.model.to(self.device)

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
        pure_arithmetic: bool = False,
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

        self.model.eval()
        self.model.to(self.device)

        results = {}

        # 1. Full model (ARB + LoRA)
        rng = random.Random(seed)
        full_correct = self._run_accuracy_test(num_samples, rng, pure_arithmetic)
        results["full"] = full_correct / num_samples
        logger.info(f"  Full (ARB+LoRA): {results['full']:.1%}")

        # 2. LoRA only (zero ARB injection)
        saved_arb = self._zero_injection()
        rng = random.Random(seed)
        lora_only_correct = self._run_accuracy_test(num_samples, rng, pure_arithmetic)
        results["lora_only"] = lora_only_correct / num_samples
        logger.info(f"  LoRA only: {results['lora_only']:.1%}")
        self._restore_injection(saved_arb)

        # 3. ARB only (zero LoRA)
        saved_lora = self._zero_lora()
        rng = random.Random(seed)
        arb_only_correct = self._run_accuracy_test(num_samples, rng, pure_arithmetic)
        results["arb_only"] = arb_only_correct / num_samples
        logger.info(f"  ARB only: {results['arb_only']:.1%}")
        self._restore_lora(saved_lora)

        # 4. Baseline (zero both)
        saved_arb = self._zero_injection()
        saved_lora = self._zero_lora()
        rng = random.Random(seed)
        baseline_correct = self._run_accuracy_test(num_samples, rng, pure_arithmetic)
        results["baseline"] = baseline_correct / num_samples
        logger.info(f"  Baseline: {results['baseline']:.1%}")
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
        self.model.eval()
        self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        total_loss = 0.0
        total_tokens = 0

        texts = eval_texts[:max_samples]

        for text in tqdm(texts, desc="Perplexity", disable=len(texts) < 10):
            encoding = self.tokenizer(
                text, truncation=True, max_length=512, return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            if input_ids.size(1) < 2:
                continue

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs["loss"]
            num_tokens = attention_mask.sum().item() - 1  # -1 for shifted labels
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
        """Run the complete evaluation suite."""
        results: dict[str, object] = {}

        logger.info("=== Exact Match Accuracy ===")
        results["exact_match"] = self.exact_match_accuracy()

        logger.info("=== Division Accuracy ===")
        results["division"] = self.division_accuracy()

        logger.info("=== Transcendental Accuracy ===")
        results["transcendental"] = self.transcendental_accuracy()

        logger.info("=== Float Arithmetic Accuracy ===")
        results["float_arithmetic"] = self.float_arithmetic_accuracy()

        logger.info("=== Multi-Step Accuracy ===")
        results["multi_step"] = self.multi_step_accuracy()

        logger.info("=== Ablation Test ===")
        results["ablation"] = self.ablation_test()

        if hasattr(self.model, "lora_head") and self.model.lora_head is not None:
            logger.info("=== Four-Way Ablation ===")
            results["four_way_ablation"] = self.four_way_ablation()

        if eval_texts:
            logger.info("=== Perplexity Test ===")
            results["perplexity"] = self.perplexity_test(eval_texts)

        return results
