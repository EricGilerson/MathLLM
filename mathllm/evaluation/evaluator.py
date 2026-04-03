"""Evaluation suite for ARB-augmented models.

Measures:
1. Exact-match accuracy on N-digit arithmetic operations
2. Ablation: zero W_proj to confirm ARB is the source of improvement
3. Perplexity regression on language benchmarks
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mathllm.config import EvalConfig
from mathllm.model.gpt2_arb import GPT2WithARB

logger = logging.getLogger(__name__)

OP_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "exp": "**"}


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
        """Extract the first number (possibly negative) from generated text."""
        text = text.strip()
        match = re.search(r"-?\d+", text)
        if match:
            return match.group()
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

    def ablation_test(self, num_samples: int = 50, seed: int = 99) -> dict[str, float]:
        """Zero out W_proj, confirm accuracy reverts to baseline.

        Saves W_proj weights, zeros them, runs accuracy test, then restores.

        Returns:
            Dict with 'ablated' and 'normal' accuracy for comparison.
        """
        import random

        rng = random.Random(seed)
        self.model.eval()
        self.model.to(self.device)

        # Normal accuracy (quick test on 2-digit addition)
        normal_correct = 0
        for _ in range(num_samples):
            a, b = _sample_operands(2, rng)
            prompt = f"{a} + {b} ="
            expected = a + b
            generated = self._generate_text(prompt, max_new_tokens=10)
            extracted = self._extract_number_from_generation(generated)
            if extracted is not None and int(extracted) == expected:
                normal_correct += 1

        # Save and zero W_proj
        saved_states = {}
        for key, arb in self.model.arbs.items():
            saved_states[key] = {
                "weight": arb.inject.projection.weight.data.clone(),
                "bias": arb.inject.projection.bias.data.clone(),
            }
            arb.inject.projection.weight.data.zero_()
            arb.inject.projection.bias.data.zero_()

        # Ablated accuracy
        rng = random.Random(seed)  # Reset for same examples
        ablated_correct = 0
        for _ in range(num_samples):
            a, b = _sample_operands(2, rng)
            prompt = f"{a} + {b} ="
            expected = a + b
            generated = self._generate_text(prompt, max_new_tokens=10)
            extracted = self._extract_number_from_generation(generated)
            if extracted is not None and int(extracted) == expected:
                ablated_correct += 1

        # Restore W_proj
        for key, arb in self.model.arbs.items():
            arb.inject.projection.weight.data.copy_(saved_states[key]["weight"])
            arb.inject.projection.bias.data.copy_(saved_states[key]["bias"])

        normal_acc = normal_correct / num_samples
        ablated_acc = ablated_correct / num_samples
        logger.info(f"Ablation: normal={normal_acc:.1%}, ablated={ablated_acc:.1%}")

        return {"normal_accuracy": normal_acc, "ablated_accuracy": ablated_acc}

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

        logger.info("=== Ablation Test ===")
        results["ablation"] = self.ablation_test()

        if eval_texts:
            logger.info("=== Perplexity Test ===")
            results["perplexity"] = self.perplexity_test(eval_texts)

        return results
