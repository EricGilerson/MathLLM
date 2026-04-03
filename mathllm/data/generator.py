"""Synthetic arithmetic data generation pipeline.

Generates three categories of training data:
1. Positive examples: arithmetic expressions in diverse formats
2. Negative examples: text with numbers but no computation
3. Edge cases: zero, identity, boundary values

Multi-step examples (2-step and 3-step) train the model to compose
operations across ARB layers, as described in the architecture.

Transcendental function examples (sin, cos, tan, exp, ln, sqrt) and
floating-point arithmetic examples train the model to compose polynomial
evaluations and Newton-Raphson iterations across multiple ARB layers.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from mathllm.config import DataConfig
from mathllm.data.negative_examples import NegativeExampleSampler
from mathllm.data.templates import (
    ADDITION_TEMPLATES,
    COSINE_TEMPLATES,
    DIVISION_EXACT_TEMPLATES,
    EXPONENTIATION_TEMPLATES,
    FLOAT_ADDITION_TEMPLATES,
    FLOAT_DIVISION_TEMPLATES,
    FLOAT_MULTIPLICATION_TEMPLATES,
    FLOAT_SUBTRACTION_TEMPLATES,
    LN_TEMPLATES,
    MULTI_STEP_2_TEMPLATES,
    MULTI_STEP_3_GENERIC_TEMPLATES,
    MULTIPLICATION_TEMPLATES,
    OP_SYMBOLS,
    SINE_TEMPLATES,
    SQRT_TEMPLATES,
    SUBTRACTION_TEMPLATES,
    TAN_TEMPLATES,
    TRANSCENDENTAL_EXP_TEMPLATES,
)


class ArithmeticDataGenerator:
    """Generate synthetic arithmetic training data with format diversity."""

    # All valid 2-step operation combos
    _OP2_COMBOS = list(MULTI_STEP_2_TEMPLATES.keys())

    # All valid 3-step operation combos (any combo of add/sub/mul)
    _OPS = ("add", "sub", "mul")
    _OP3_COMBOS = [
        (o1, o2, o3)
        for o1 in ("add", "sub", "mul")
        for o2 in ("add", "sub", "mul")
        for o3 in ("add", "sub", "mul")
    ]

    def __init__(self, config: DataConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.negative_sampler = NegativeExampleSampler(seed=config.seed)
        self.max_value = config.max_value

    def _sample_operands(self, num_digits: int | None = None) -> tuple[int, int]:
        """Sample two operands with controlled digit count."""
        if num_digits is None:
            num_digits = self.rng.randint(1, self.config.max_digits)
        low = 10 ** (num_digits - 1) if num_digits > 1 else 0
        high = 10**num_digits - 1
        return self.rng.randint(low, high), self.rng.randint(low, high)

    def _generate_addition(self) -> str | None:
        a, b = self._sample_operands()
        result = a + b
        if abs(result) > self.max_value:
            return None
        template = self.rng.choice(ADDITION_TEMPLATES)
        return template.format(a=a, b=b, result=result)

    def _generate_subtraction(self) -> str | None:
        a, b = self._sample_operands()
        result = a - b
        template = self.rng.choice(SUBTRACTION_TEMPLATES)
        return template.format(a=a, b=b, result=result)

    def _generate_multiplication(self) -> str | None:
        # Use smaller operands to keep product in range
        d1 = self.rng.randint(1, min(5, self.config.max_digits))
        d2 = self.rng.randint(1, min(5, self.config.max_digits))
        low1 = 10 ** (d1 - 1) if d1 > 1 else 0
        low2 = 10 ** (d2 - 1) if d2 > 1 else 0
        a = self.rng.randint(low1, 10**d1 - 1)
        b = self.rng.randint(low2, 10**d2 - 1)
        result = a * b
        if abs(result) > self.max_value:
            return None
        template = self.rng.choice(MULTIPLICATION_TEMPLATES)
        return template.format(a=a, b=b, result=result)

    def _generate_exponentiation(self) -> str | None:
        b = self.rng.randint(0, 15)
        if b == 0:
            a = self.rng.randint(1, 999)
        else:
            a_max = int(self.max_value ** (1.0 / max(b, 1)))
            a = self.rng.randint(1, max(a_max, 2))
        result = a**b
        if result > self.max_value:
            return None
        template = self.rng.choice(EXPONENTIATION_TEMPLATES)
        return template.format(a=a, b=b, result=result)

    def _generate_exact_division(self) -> str | None:
        """Generate exact division (no remainder)."""
        b = self.rng.randint(1, 999)
        quotient = self.rng.randint(1, max(1, self.max_value // max(b, 1)))
        a = b * quotient
        if a > self.max_value:
            return None
        template = self.rng.choice(DIVISION_EXACT_TEMPLATES)
        return template.format(a=a, b=b, result=quotient)

    # ------------------------------------------------------------------
    # Transcendental functions
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_float(x: float, decimals: int = 6) -> str:
        """Format a float to a fixed number of decimal places."""
        return f"{x:.{decimals}f}"

    def _generate_sine(self) -> str | None:
        """Generate sin(x) = result with x in [0, 2*pi]."""
        x = self.rng.uniform(0, 2 * math.pi)
        result = math.sin(x)
        template = self.rng.choice(SINE_TEMPLATES)
        return template.format(x=self._fmt_float(x), result=self._fmt_float(result))

    def _generate_cosine(self) -> str | None:
        """Generate cos(x) = result with x in [0, 2*pi]."""
        x = self.rng.uniform(0, 2 * math.pi)
        result = math.cos(x)
        template = self.rng.choice(COSINE_TEMPLATES)
        return template.format(x=self._fmt_float(x), result=self._fmt_float(result))

    def _generate_tan(self) -> str | None:
        """Generate tan(x) = result, avoiding near-singularities."""
        # Sample from safe regions (avoid pi/2 + n*pi)
        x = self.rng.uniform(-1.4, 1.4)  # within (-pi/2, pi/2)
        result = math.tan(x)
        if abs(result) > 1000:
            return None
        template = self.rng.choice(TAN_TEMPLATES)
        return template.format(x=self._fmt_float(x), result=self._fmt_float(result))

    def _generate_transcendental_exp(self) -> str | None:
        """Generate exp(x) = result with x in [-5, 20]."""
        x = self.rng.uniform(-5, 20)
        result = math.exp(x)
        if result > 1e9:
            return None
        template = self.rng.choice(TRANSCENDENTAL_EXP_TEMPLATES)
        return template.format(x=self._fmt_float(x), result=self._fmt_float(result))

    def _generate_ln(self) -> str | None:
        """Generate ln(x) = result with x in (0, 10^6]."""
        x = self.rng.uniform(0.01, 1_000_000)
        result = math.log(x)
        template = self.rng.choice(LN_TEMPLATES)
        return template.format(x=self._fmt_float(x), result=self._fmt_float(result))

    def _generate_sqrt(self) -> str | None:
        """Generate sqrt(x) = result with x in [0, 10^6]."""
        x = self.rng.uniform(0, 1_000_000)
        result = math.sqrt(x)
        template = self.rng.choice(SQRT_TEMPLATES)
        return template.format(x=self._fmt_float(x), result=self._fmt_float(result))

    # ------------------------------------------------------------------
    # Floating-point arithmetic
    # ------------------------------------------------------------------

    def _sample_float_operands(self) -> tuple[float, float]:
        """Sample two float operands with 1-6 decimal places."""
        decimals = self.rng.randint(1, 6)
        scale = 10 ** decimals
        a = self.rng.randint(1, int(min(self.max_value, 999999) * scale)) / scale
        b = self.rng.randint(1, int(min(self.max_value, 999999) * scale)) / scale
        return round(a, decimals), round(b, decimals)

    def _generate_float_addition(self) -> str | None:
        a, b = self._sample_float_operands()
        result = a + b
        if abs(result) > self.max_value:
            return None
        template = self.rng.choice(FLOAT_ADDITION_TEMPLATES)
        return template.format(a=a, b=b, result=self._fmt_float(result))

    def _generate_float_subtraction(self) -> str | None:
        a, b = self._sample_float_operands()
        result = a - b
        template = self.rng.choice(FLOAT_SUBTRACTION_TEMPLATES)
        return template.format(a=a, b=b, result=self._fmt_float(result))

    def _generate_float_multiplication(self) -> str | None:
        # Smaller operands to avoid overflow
        decimals = self.rng.randint(1, 4)
        scale = 10 ** decimals
        a = self.rng.randint(1, int(999 * scale)) / scale
        b = self.rng.randint(1, int(999 * scale)) / scale
        a, b = round(a, decimals), round(b, decimals)
        result = a * b
        if abs(result) > self.max_value:
            return None
        template = self.rng.choice(FLOAT_MULTIPLICATION_TEMPLATES)
        return template.format(a=a, b=b, result=self._fmt_float(result))

    def _generate_float_division(self) -> str | None:
        a, b = self._sample_float_operands()
        if b == 0:
            return None
        result = a / b
        if abs(result) > self.max_value:
            return None
        template = self.rng.choice(FLOAT_DIVISION_TEMPLATES)
        return template.format(a=a, b=b, result=self._fmt_float(result))

    # ------------------------------------------------------------------
    # Helpers for multi-step
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_op(x: int, y: int, op: str) -> int | None:
        """Apply a single operation. Returns None if overflow would occur."""
        if op == "add":
            return x + y
        elif op == "sub":
            return x - y
        elif op == "mul":
            return x * y
        return None

    # ------------------------------------------------------------------
    # Multi-step: 2-step operation chains
    # ------------------------------------------------------------------

    def _generate_multi_step_2(self) -> str | None:
        """Generate a two-step arithmetic expression.

        Picks a random (op1, op2) combo, generates matching operands,
        then selects a template from the correct combo bucket so the
        text always matches the actual computation.
        """
        op1, op2 = self.rng.choice(self._OP2_COMBOS)

        # Smaller operands for mul to avoid overflow
        d = self.rng.randint(1, 3 if op1 == "mul" else 4)
        a1, b1 = self._sample_operands(d)

        r1 = self._apply_op(a1, b1, op1)
        if r1 is None or abs(r1) > self.max_value:
            return None

        # Second operand -- smaller for mul
        b2_max = 100 if op2 == "mul" else 999
        b2 = self.rng.randint(1, b2_max)

        result = self._apply_op(r1, b2, op2)
        if result is None or abs(result) > self.max_value:
            return None

        templates = MULTI_STEP_2_TEMPLATES[(op1, op2)]
        template = self.rng.choice(templates)
        try:
            return template.format(a1=a1, b1=b1, r1=r1, b2=b2, result=result)
        except (KeyError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Multi-step: 3-step operation chains
    # ------------------------------------------------------------------

    def _generate_multi_step_3(self) -> str | None:
        """Generate a three-step arithmetic expression.

        Uses generic templates with operation symbols filled in to match
        the actual operations. Shows all intermediate results.
        """
        op1, op2, op3 = self.rng.choice(self._OP3_COMBOS)

        # Small operands to avoid overflow across 3 ops
        d = self.rng.randint(1, 2 if op1 == "mul" else 3)
        a1, b1 = self._sample_operands(d)

        r1 = self._apply_op(a1, b1, op1)
        if r1 is None or abs(r1) > self.max_value:
            return None

        b2_max = 50 if op2 == "mul" else 500
        b2 = self.rng.randint(1, b2_max)

        r2 = self._apply_op(r1, b2, op2)
        if r2 is None or abs(r2) > self.max_value:
            return None

        b3_max = 50 if op3 == "mul" else 500
        b3 = self.rng.randint(1, b3_max)

        result = self._apply_op(r2, b3, op3)
        if result is None or abs(result) > self.max_value:
            return None

        template = self.rng.choice(MULTI_STEP_3_GENERIC_TEMPLATES)
        try:
            return template.format(
                a1=a1, b1=b1, r1=r1, b2=b2, r2=r2, b3=b3, result=result,
                sym1=OP_SYMBOLS[op1], sym2=OP_SYMBOLS[op2], sym3=OP_SYMBOLS[op3],
            )
        except (KeyError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def _generate_edge_case(self) -> str:
        """Generate edge case examples."""
        case_type = self.rng.choice([
            "zero_add", "zero_mul", "identity_mul", "self_sub",
            "power_zero", "power_one", "square", "small_add",
            "boundary_add", "negative_result", "one_digit_all_ops",
            "double_zero", "large_identity", "cube",
            "consecutive_add", "near_overflow_mul", "power_two",
            "small_mul", "commutative_pair", "large_sub_to_zero",
            "divide_by_one", "divide_self", "square_root_exact",
            "sin_zero", "cos_zero", "sin_pi_half", "cos_pi",
            "exp_zero", "ln_one", "sqrt_zero", "sqrt_one",
        ])

        if case_type == "zero_add":
            a = self.rng.randint(0, self.max_value)
            template = self.rng.choice(ADDITION_TEMPLATES)
            return template.format(a=a, b=0, result=a)
        elif case_type == "zero_mul":
            a = self.rng.randint(0, self.max_value)
            template = self.rng.choice(MULTIPLICATION_TEMPLATES)
            return template.format(a=a, b=0, result=0)
        elif case_type == "identity_mul":
            a = self.rng.randint(0, self.max_value)
            template = self.rng.choice(MULTIPLICATION_TEMPLATES)
            return template.format(a=a, b=1, result=a)
        elif case_type == "self_sub":
            a = self.rng.randint(0, self.max_value)
            template = self.rng.choice(SUBTRACTION_TEMPLATES)
            return template.format(a=a, b=a, result=0)
        elif case_type == "power_zero":
            a = self.rng.randint(1, 999)
            template = self.rng.choice(EXPONENTIATION_TEMPLATES)
            return template.format(a=a, b=0, result=1)
        elif case_type == "power_one":
            a = self.rng.randint(1, 999)
            template = self.rng.choice(EXPONENTIATION_TEMPLATES)
            return template.format(a=a, b=1, result=a)
        elif case_type == "square":
            a = self.rng.randint(1, 31622)  # sqrt(10^9) ~ 31622
            template = self.rng.choice(EXPONENTIATION_TEMPLATES)
            return template.format(a=a, b=2, result=a * a)
        elif case_type == "cube":
            a = self.rng.randint(1, 999)  # cbrt(10^9) ~ 999
            template = self.rng.choice(EXPONENTIATION_TEMPLATES)
            return template.format(a=a, b=3, result=a * a * a)
        elif case_type == "small_add":
            a = self.rng.randint(0, 9)
            b = self.rng.randint(0, 9)
            template = self.rng.choice(ADDITION_TEMPLATES)
            return template.format(a=a, b=b, result=a + b)
        elif case_type == "one_digit_all_ops":
            a = self.rng.randint(1, 9)
            b = self.rng.randint(1, 9)
            op = self.rng.choice(["add", "sub", "mul"])
            if op == "add":
                template = self.rng.choice(ADDITION_TEMPLATES)
                return template.format(a=a, b=b, result=a + b)
            elif op == "sub":
                template = self.rng.choice(SUBTRACTION_TEMPLATES)
                return template.format(a=a, b=b, result=a - b)
            else:
                template = self.rng.choice(MULTIPLICATION_TEMPLATES)
                return template.format(a=a, b=b, result=a * b)
        elif case_type == "double_zero":
            template = self.rng.choice(ADDITION_TEMPLATES)
            return template.format(a=0, b=0, result=0)
        elif case_type == "large_identity":
            a = self.max_value - self.rng.randint(0, 1000)
            template = self.rng.choice(MULTIPLICATION_TEMPLATES)
            return template.format(a=a, b=1, result=a)
        elif case_type == "boundary_add":
            a = self.max_value - self.rng.randint(1, 1000)
            b = self.rng.randint(1, 999)
            if a + b <= self.max_value:
                template = self.rng.choice(ADDITION_TEMPLATES)
                return template.format(a=a, b=b, result=a + b)
            return self._generate_edge_case()
        elif case_type == "negative_result":
            b = self.rng.randint(1, 999)
            a = self.rng.randint(0, b - 1)
            template = self.rng.choice(SUBTRACTION_TEMPLATES)
            return template.format(a=a, b=b, result=a - b)
        elif case_type == "consecutive_add":
            n = self.rng.randint(0, self.max_value // 2 - 1)
            a, b = n, n + 1
            template = self.rng.choice(ADDITION_TEMPLATES)
            return template.format(a=a, b=b, result=a + b)
        elif case_type == "near_overflow_mul":
            # Product barely under max_value
            b = self.rng.randint(2, 999)
            quotient = self.max_value // b
            a = self.rng.randint(max(1, quotient - 100), quotient)
            result = a * b
            if result <= self.max_value:
                template = self.rng.choice(MULTIPLICATION_TEMPLATES)
                return template.format(a=a, b=b, result=result)
            return self._generate_edge_case()
        elif case_type == "power_two":
            n = self.rng.randint(0, 29)  # 2^29 < 10^9
            result = 2 ** n
            template = self.rng.choice(EXPONENTIATION_TEMPLATES)
            return template.format(a=2, b=n, result=result)
        elif case_type == "small_mul":
            a = self.rng.randint(1, 9)
            b = self.rng.randint(1, 9)
            template = self.rng.choice(MULTIPLICATION_TEMPLATES)
            return template.format(a=a, b=b, result=a * b)
        elif case_type == "commutative_pair":
            a = self.rng.randint(1, 999)
            b = self.rng.randint(1, 999)
            result = a + b
            if result <= self.max_value:
                template = self.rng.choice(ADDITION_TEMPLATES)
                return template.format(a=b, b=a, result=result)
            return self._generate_edge_case()
        elif case_type == "large_sub_to_zero":
            a = self.rng.randint(1000, self.max_value)
            b = a - self.rng.randint(0, 9)
            template = self.rng.choice(SUBTRACTION_TEMPLATES)
            return template.format(a=a, b=b, result=a - b)
        elif case_type == "divide_by_one":
            a = self.rng.randint(1, self.max_value)
            template = self.rng.choice(DIVISION_EXACT_TEMPLATES)
            return template.format(a=a, b=1, result=a)
        elif case_type == "divide_self":
            a = self.rng.randint(1, self.max_value)
            template = self.rng.choice(DIVISION_EXACT_TEMPLATES)
            return template.format(a=a, b=a, result=1)
        elif case_type == "sin_zero":
            template = self.rng.choice(SINE_TEMPLATES)
            return template.format(x="0.000000", result="0.000000")
        elif case_type == "cos_zero":
            template = self.rng.choice(COSINE_TEMPLATES)
            return template.format(x="0.000000", result="1.000000")
        elif case_type == "sin_pi_half":
            template = self.rng.choice(SINE_TEMPLATES)
            return template.format(x=self._fmt_float(math.pi / 2), result="1.000000")
        elif case_type == "cos_pi":
            template = self.rng.choice(COSINE_TEMPLATES)
            return template.format(x=self._fmt_float(math.pi), result="-1.000000")
        elif case_type == "exp_zero":
            template = self.rng.choice(TRANSCENDENTAL_EXP_TEMPLATES)
            return template.format(x="0.000000", result="1.000000")
        elif case_type == "ln_one":
            template = self.rng.choice(LN_TEMPLATES)
            return template.format(x="1.000000", result="0.000000")
        elif case_type == "sqrt_zero":
            template = self.rng.choice(SQRT_TEMPLATES)
            return template.format(x="0.000000", result="0.000000")
        elif case_type == "sqrt_one":
            template = self.rng.choice(SQRT_TEMPLATES)
            return template.format(x="1.000000", result="1.000000")
        else:  # square_root_exact
            a = self.rng.randint(1, 31622)  # sqrt(10^9) ~ 31622
            product = a * a
            template = self.rng.choice(MULTIPLICATION_TEMPLATES)
            return template.format(a=a, b=a, result=product)

    # ------------------------------------------------------------------
    # Top-level generation
    # ------------------------------------------------------------------

    def _generate_positive_example(self) -> str | None:
        """Generate one positive arithmetic example."""
        # Multi-step gets 2x weight; transcendentals and floats each get 1x
        op = self.rng.choice([
            "add", "sub", "mul", "exp", "div",
            "sin", "cos", "tan", "texp", "ln", "sqrt",
            "float_add", "float_sub", "float_mul", "float_div",
            "multi_step_2", "multi_step_2", "multi_step_3",
        ])
        generators: dict[str, Any] = {
            "add": self._generate_addition,
            "sub": self._generate_subtraction,
            "mul": self._generate_multiplication,
            "exp": self._generate_exponentiation,
            "div": self._generate_exact_division,
            "sin": self._generate_sine,
            "cos": self._generate_cosine,
            "tan": self._generate_tan,
            "texp": self._generate_transcendental_exp,
            "ln": self._generate_ln,
            "sqrt": self._generate_sqrt,
            "float_add": self._generate_float_addition,
            "float_sub": self._generate_float_subtraction,
            "float_mul": self._generate_float_multiplication,
            "float_div": self._generate_float_division,
            "multi_step_2": self._generate_multi_step_2,
            "multi_step_3": self._generate_multi_step_3,
        }
        return generators[op]()

    def generate_dataset(self) -> list[str]:
        """Generate the complete training dataset."""
        examples: list[str] = []

        # Positive examples
        attempts = 0
        max_attempts = self.config.num_positive * 5
        while len(examples) < self.config.num_positive and attempts < max_attempts:
            ex = self._generate_positive_example()
            if ex is not None:
                examples.append(ex)
            attempts += 1

        # Negative examples
        negatives = self.negative_sampler.sample(self.config.num_negative)
        examples.extend(negatives)

        # Edge cases
        for _ in range(self.config.num_edge_cases):
            examples.append(self._generate_edge_case())

        self.rng.shuffle(examples)
        return examples

    def save_dataset(self, examples: list[str], output_dir: str | Path) -> Path:
        """Save dataset as JSONL file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "train.jsonl"

        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps({"text": ex}) + "\n")

        return output_path
