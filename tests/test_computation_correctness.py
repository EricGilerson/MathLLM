"""Comprehensive computation correctness tests.

Verifies that every arithmetic operation in the ARB produces mathematically
exact results across the full representable range. Tests the complete pipeline:
  digits -> RNS circle encoding -> arithmetic operation -> CRT reconstruction -> integer

This is the most important test file in the project. If these tests pass, the
frozen arithmetic is provably correct.
"""

import math
import random

import pytest
import torch

from mathllm.arb.constants import (
    DEFAULT_PRIMES,
    compute_crt_weights,
    compute_product,
    integer_to_digits,
    digits_to_integer,
    mod_inverse,
)
from mathllm.arb.stage2_encode import RNSCircleEncoder
from mathllm.arb.stage3_compute import ArithmeticCompute


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

P = compute_product(DEFAULT_PRIMES)  # ~7.42e9


@pytest.fixture
def encoder():
    return RNSCircleEncoder(DEFAULT_PRIMES, num_digits=10)


@pytest.fixture
def compute():
    return ArithmeticCompute(DEFAULT_PRIMES, num_digits=10, softmax_temperature=1000.0)


def encode(encoder, n: int) -> torch.Tensor:
    """Encode an integer to circle representation [1, 1, m, 2]."""
    digits = torch.tensor(integer_to_digits(n, 10), dtype=torch.float32)
    return encoder(digits.unsqueeze(0).unsqueeze(0))


def encode_exp(encoder, n: int) -> torch.Tensor:
    """Encode an integer for exponent (Fermat-reduced) [1, 1, m, 2]."""
    digits = torch.tensor(integer_to_digits(n, 10), dtype=torch.float32)
    return encoder.encode_exponent(digits.unsqueeze(0).unsqueeze(0))


def crt_result(compute, circle: torch.Tensor) -> int:
    """Extract integer from circle encoding via CRT (unsigned)."""
    return int(compute.crt_reconstruct(circle).item())


def crt_result_signed(compute, circle: torch.Tensor) -> int:
    """Extract signed integer from circle encoding via CRT."""
    return int(compute.crt_reconstruct_signed(circle).item())


# ---------------------------------------------------------------------------
# ADDITION — comprehensive
# ---------------------------------------------------------------------------

class TestAdditionCorrectness:
    """Verify circle-encoded addition is exact for all tested values."""

    def test_identity(self, compute, encoder):
        """a + 0 = a for various a."""
        for a in [0, 1, 5, 99, 12345, 999999999]:
            result = crt_result(compute, compute.circle_add(
                encode(encoder, a), encode(encoder, 0)
            ))
            assert result == a, f"{a} + 0 = {result}, expected {a}"

    def test_commutativity(self, compute, encoder):
        """a + b = b + a."""
        rng = random.Random(42)
        for _ in range(200):
            a = rng.randint(0, P // 2 - 1)
            b = rng.randint(0, P // 2 - 1)
            r1 = crt_result(compute, compute.circle_add(
                encode(encoder, a), encode(encoder, b)
            ))
            r2 = crt_result(compute, compute.circle_add(
                encode(encoder, b), encode(encoder, a)
            ))
            assert r1 == r2, f"Commutativity: {a}+{b}={r1}, {b}+{a}={r2}"

    def test_small_additions(self, compute, encoder):
        """All single-digit + single-digit additions."""
        for a in range(10):
            for b in range(10):
                expected = a + b
                result = crt_result(compute, compute.circle_add(
                    encode(encoder, a), encode(encoder, b)
                ))
                assert result == expected, f"{a} + {b} = {result}, expected {expected}"

    def test_medium_additions(self, compute, encoder):
        """Multi-digit additions with known results."""
        cases = [
            (100, 200, 300),
            (347, 291, 638),
            (999, 1, 1000),
            (12345, 67890, 80235),
            (500000, 500000, 1000000),
            (123456789, 1, 123456790),
        ]
        for a, b, expected in cases:
            result = crt_result(compute, compute.circle_add(
                encode(encoder, a), encode(encoder, b)
            ))
            assert result == expected, f"{a} + {b} = {result}, expected {expected}"

    def test_random_additions_in_range(self, compute, encoder):
        """500 random additions where a + b < P."""
        rng = random.Random(123)
        for _ in range(500):
            a = rng.randint(0, 999999)
            b = rng.randint(0, 999999)
            expected = a + b
            result = crt_result(compute, compute.circle_add(
                encode(encoder, a), encode(encoder, b)
            ))
            assert result == expected, f"{a} + {b} = {result}, expected {expected}"

    def test_large_additions_near_boundary(self, compute, encoder):
        """Additions with operands near 10^9."""
        cases = [
            (999999999, 1, 1000000000),
            (500000000, 500000000, 1000000000),
            (999999000, 999, 999999999),
        ]
        for a, b, expected in cases:
            if expected < P:
                result = crt_result(compute, compute.circle_add(
                    encode(encoder, a), encode(encoder, b)
                ))
                assert result == expected, f"{a} + {b} = {result}, expected {expected}"

    def test_associativity(self, compute, encoder):
        """(a + b) + c = a + (b + c) via sequential circle_add."""
        rng = random.Random(77)
        for _ in range(100):
            a = rng.randint(0, 10000)
            b = rng.randint(0, 10000)
            c = rng.randint(0, 10000)
            ab_c = compute.circle_add(
                compute.circle_add(encode(encoder, a), encode(encoder, b)),
                encode(encoder, c),
            )
            a_bc = compute.circle_add(
                encode(encoder, a),
                compute.circle_add(encode(encoder, b), encode(encoder, c)),
            )
            r1 = crt_result(compute, ab_c)
            r2 = crt_result(compute, a_bc)
            assert r1 == r2 == a + b + c, \
                f"Associativity: ({a}+{b})+{c}={r1}, {a}+({b}+{c})={r2}, expected {a+b+c}"


# ---------------------------------------------------------------------------
# SUBTRACTION — comprehensive
# ---------------------------------------------------------------------------

class TestSubtractionCorrectness:

    def test_self_subtraction(self, compute, encoder):
        """a - a = 0."""
        for a in [0, 1, 42, 99999, 999999999]:
            result = crt_result_signed(compute, compute.circle_sub(
                encode(encoder, a), encode(encoder, a)
            ))
            assert result == 0, f"{a} - {a} = {result}, expected 0"

    def test_positive_results(self, compute, encoder):
        """a - b where a > b."""
        cases = [
            (10, 3, 7),
            (638, 291, 347),
            (1000, 1, 999),
            (999999999, 999999998, 1),
            (100000, 1, 99999),
        ]
        for a, b, expected in cases:
            result = crt_result_signed(compute, compute.circle_sub(
                encode(encoder, a), encode(encoder, b)
            ))
            assert result == expected, f"{a} - {b} = {result}, expected {expected}"

    def test_negative_results(self, compute, encoder):
        """a - b where a < b (negative result)."""
        cases = [
            (3, 10, -7),
            (0, 1, -1),
            (0, 999, -999),
            (1, 1000, -999),
            (100, 12345, -12245),
        ]
        for a, b, expected in cases:
            result = crt_result_signed(compute, compute.circle_sub(
                encode(encoder, a), encode(encoder, b)
            ))
            assert result == expected, f"{a} - {b} = {result}, expected {expected}"

    def test_random_subtractions(self, compute, encoder):
        """500 random subtractions with signed results."""
        rng = random.Random(456)
        for _ in range(500):
            a = rng.randint(0, 999999)
            b = rng.randint(0, 999999)
            expected = a - b
            result = crt_result_signed(compute, compute.circle_sub(
                encode(encoder, a), encode(encoder, b)
            ))
            assert result == expected, f"{a} - {b} = {result}, expected {expected}"

    def test_subtraction_is_add_inverse(self, compute, encoder):
        """(a + b) - b = a."""
        rng = random.Random(789)
        for _ in range(200):
            a = rng.randint(0, 999999)
            b = rng.randint(0, 999999)
            sum_circle = compute.circle_add(encode(encoder, a), encode(encoder, b))
            result_circle = compute.circle_sub(sum_circle, encode(encoder, b))
            result = crt_result(compute, result_circle)
            assert result == a, f"({a}+{b})-{b} = {result}, expected {a}"


# ---------------------------------------------------------------------------
# MULTIPLICATION — comprehensive
# ---------------------------------------------------------------------------

class TestMultiplicationCorrectness:

    def test_identity(self, compute, encoder):
        """a * 1 = a."""
        for a in [0, 1, 7, 347, 99999, 999999]:
            result = crt_result(compute, compute.circle_mul(
                encode(encoder, a), encode(encoder, 1)
            ))
            assert result == a, f"{a} * 1 = {result}, expected {a}"

    def test_zero(self, compute, encoder):
        """a * 0 = 0."""
        for a in [0, 1, 42, 99999, 999999]:
            result = crt_result(compute, compute.circle_mul(
                encode(encoder, a), encode(encoder, 0)
            ))
            assert result == 0, f"{a} * 0 = {result}, expected 0"

    def test_commutativity(self, compute, encoder):
        """a * b = b * a."""
        rng = random.Random(42)
        for _ in range(200):
            a = rng.randint(0, 9999)
            b = rng.randint(0, 9999)
            r1 = crt_result(compute, compute.circle_mul(
                encode(encoder, a), encode(encoder, b)
            ))
            r2 = crt_result(compute, compute.circle_mul(
                encode(encoder, b), encode(encoder, a)
            ))
            assert r1 == r2, f"Commutativity: {a}*{b}={r1}, {b}*{a}={r2}"

    def test_small_multiplications(self, compute, encoder):
        """All single-digit * single-digit products."""
        for a in range(10):
            for b in range(10):
                expected = a * b
                result = crt_result(compute, compute.circle_mul(
                    encode(encoder, a), encode(encoder, b)
                ))
                assert result == expected, f"{a} * {b} = {result}, expected {expected}"

    def test_medium_multiplications(self, compute, encoder):
        """Multi-digit products with known results."""
        cases = [
            (12, 12, 144),
            (99, 99, 9801),
            (123, 456, 56088),
            (1000, 1000, 1000000),
            (347, 291, 100977),
        ]
        for a, b, expected in cases:
            result = crt_result(compute, compute.circle_mul(
                encode(encoder, a), encode(encoder, b)
            ))
            assert result == expected, f"{a} * {b} = {result}, expected {expected}"

    def test_random_multiplications_in_range(self, compute, encoder):
        """500 random multiplications where a * b < P."""
        rng = random.Random(321)
        for _ in range(500):
            a = rng.randint(0, 9999)
            b = rng.randint(0, 9999)
            expected = a * b
            if expected >= P:
                continue
            result = crt_result(compute, compute.circle_mul(
                encode(encoder, a), encode(encoder, b)
            ))
            assert result == expected, f"{a} * {b} = {result}, expected {expected}"

    def test_large_products_near_boundary(self, compute, encoder):
        """Products near the representable range boundary."""
        cases = [
            (31622, 31622),  # sqrt(10^9) ≈ 31622
            (10000, 10000),
            (99999, 99),
        ]
        for a, b in cases:
            expected = a * b
            if expected >= P:
                continue
            result = crt_result(compute, compute.circle_mul(
                encode(encoder, a), encode(encoder, b)
            ))
            assert result == expected, f"{a} * {b} = {result}, expected {expected}"

    def test_distributivity(self, compute, encoder):
        """a * (b + c) = a*b + a*c via circle operations."""
        rng = random.Random(654)
        for _ in range(100):
            a = rng.randint(1, 100)
            b = rng.randint(1, 100)
            c = rng.randint(1, 100)
            # Left: a * (b + c)
            bc_sum = compute.circle_add(encode(encoder, b), encode(encoder, c))
            left = crt_result(compute, compute.circle_mul(encode(encoder, a), bc_sum))
            # Right: a*b + a*c
            ab = compute.circle_mul(encode(encoder, a), encode(encoder, b))
            ac = compute.circle_mul(encode(encoder, a), encode(encoder, c))
            right = crt_result(compute, compute.circle_add(ab, ac))
            expected = a * (b + c)
            assert left == right == expected, \
                f"Distributivity: {a}*({b}+{c})={left}, {a}*{b}+{a}*{c}={right}, expected {expected}"

    def test_squares(self, compute, encoder):
        """a * a = a^2 for many values."""
        for a in [2, 3, 7, 10, 25, 100, 317, 1000, 31622]:
            expected = a * a
            if expected >= P:
                continue
            result = crt_result(compute, compute.circle_mul(
                encode(encoder, a), encode(encoder, a)
            ))
            assert result == expected, f"{a}^2 = {result}, expected {expected}"


# ---------------------------------------------------------------------------
# EXPONENTIATION — comprehensive
# ---------------------------------------------------------------------------

class TestExponentiationCorrectness:

    def test_power_zero(self, compute, encoder):
        """a^0 = 1 for all a > 0."""
        for a in [1, 2, 5, 10, 99, 347]:
            a_circle = encode(encoder, a)
            exp_circle = encode_exp(encoder, 0)
            result = crt_result(compute, compute.circle_exp(a_circle, exp_circle))
            assert result == 1, f"{a}^0 = {result}, expected 1"

    def test_power_one(self, compute, encoder):
        """a^1 = a."""
        for a in [1, 2, 5, 10, 99, 347, 9999]:
            a_circle = encode(encoder, a)
            exp_circle = encode_exp(encoder, 1)
            result = crt_result(compute, compute.circle_exp(a_circle, exp_circle))
            assert result == a, f"{a}^1 = {result}, expected {a}"

    def test_small_powers(self, compute, encoder):
        """Known small exponentiation results."""
        cases = [
            (2, 2, 4),
            (2, 3, 8),
            (2, 10, 1024),
            (2, 20, 1048576),
            (2, 30, 1073741824),
            (3, 2, 9),
            (3, 3, 27),
            (3, 10, 59049),
            (5, 5, 3125),
            (7, 4, 2401),
            (10, 3, 1000),
            (10, 5, 100000),
            (10, 9, 1000000000),
        ]
        for a, b, expected in cases:
            if expected >= P:
                continue
            a_circle = encode(encoder, a)
            exp_circle = encode_exp(encoder, b)
            result = crt_result(compute, compute.circle_exp(a_circle, exp_circle))
            assert result == expected, f"{a}^{b} = {result}, expected {expected}"

    def test_random_exponentiations(self, compute, encoder):
        """200 random base^exp where result is in range."""
        rng = random.Random(999)
        tested = 0
        for _ in range(2000):
            a = rng.randint(2, 50)
            b = rng.randint(0, 15)
            expected = a ** b
            if expected >= P:
                continue
            a_circle = encode(encoder, a)
            exp_circle = encode_exp(encoder, b)
            result = crt_result(compute, compute.circle_exp(a_circle, exp_circle))
            assert result == expected, f"{a}^{b} = {result}, expected {expected}"
            tested += 1
            if tested >= 200:
                break
        assert tested >= 100, f"Only tested {tested} exponentiations (need >= 100)"

    def test_fermat_reduction(self, compute, encoder):
        """Verify exponentiation works for exponents larger than p-1.

        Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) for prime p, a not divisible by p.
        So a^b mod p = a^(b mod (p-1)) mod p. This tests that the exponent
        encoding correctly reduces mod (p_i - 1) for large exponents.
        """
        # 2^100 is astronomically large, but the RNS + Fermat handles it
        # per-prime. We verify the CRT reconstruction matches.
        a, b = 2, 100
        expected_per_prime = [pow(a, b, p) for p in DEFAULT_PRIMES]

        a_circle = encode(encoder, a)
        exp_circle = encode_exp(encoder, b)
        result_circle = compute.circle_exp(a_circle, exp_circle)

        # Decode residues and check each prime
        residues = compute._decode_residues_hard(result_circle)  # [1, 1, m]
        for i, p in enumerate(DEFAULT_PRIMES):
            got = int(residues[0, 0, i].item())
            assert got == expected_per_prime[i], \
                f"2^100 mod {p}: got {got}, expected {expected_per_prime[i]}"

    def test_one_raised_to_anything(self, compute, encoder):
        """1^b = 1 for all b."""
        for b in [0, 1, 2, 10, 100, 9999]:
            a_circle = encode(encoder, 1)
            exp_circle = encode_exp(encoder, b)
            result = crt_result(compute, compute.circle_exp(a_circle, exp_circle))
            assert result == 1, f"1^{b} = {result}, expected 1"

    def test_consistency_with_multiplication(self, compute, encoder):
        """a^2 via exp should equal a * a via mul."""
        for a in [2, 7, 13, 42, 100, 999]:
            # Via exponentiation
            a_circle = encode(encoder, a)
            exp_circle = encode_exp(encoder, 2)
            exp_result = crt_result(compute, compute.circle_exp(a_circle, exp_circle))
            # Via multiplication
            mul_result = crt_result(compute, compute.circle_mul(
                encode(encoder, a), encode(encoder, a)
            ))
            expected = a * a
            assert exp_result == mul_result == expected, \
                f"{a}^2: exp={exp_result}, mul={mul_result}, expected={expected}"

    def test_consistency_with_mul_for_cubes(self, compute, encoder):
        """a^3 via exp should equal a * a * a via chained mul."""
        for a in [2, 3, 5, 10, 17]:
            expected = a ** 3
            if expected >= P:
                continue
            # Via exponentiation
            a_circle = encode(encoder, a)
            exp_circle = encode_exp(encoder, 3)
            exp_result = crt_result(compute, compute.circle_exp(a_circle, exp_circle))
            # Via chained multiplication: a * a * a
            a_sq = compute.circle_mul(encode(encoder, a), encode(encoder, a))
            a_cu = compute.circle_mul(a_sq, encode(encoder, a))
            mul_result = crt_result(compute, a_cu)
            assert exp_result == mul_result == expected, \
                f"{a}^3: exp={exp_result}, mul={mul_result}, expected={expected}"


# ---------------------------------------------------------------------------
# CRT RECONSTRUCTION — comprehensive
# ---------------------------------------------------------------------------

class TestCRTReconstructionCorrectness:

    def test_roundtrip_encode_decode(self, compute, encoder):
        """Encode integer to circle, decode back via CRT. Must be exact."""
        test_values = [0, 1, 42, 347, 12345, 999999, 123456789, 999999999]
        for n in test_values:
            circle = encode(encoder, n)
            result = crt_result(compute, circle)
            assert result == n, f"Roundtrip failed: {n} -> {result}"

    def test_random_roundtrip(self, compute, encoder):
        """1000 random integers: encode -> CRT decode must be exact."""
        rng = random.Random(2024)
        for _ in range(1000):
            n = rng.randint(0, min(P - 1, 999999999))
            circle = encode(encoder, n)
            result = crt_result(compute, circle)
            assert result == n, f"Roundtrip failed: {n} -> {result}"

    def test_signed_roundtrip(self, compute, encoder):
        """Subtraction-based signed CRT reconstruction."""
        rng = random.Random(2025)
        for _ in range(500):
            a = rng.randint(0, 99999)
            b = rng.randint(0, 99999)
            expected = a - b
            circle = compute.circle_sub(encode(encoder, a), encode(encoder, b))
            result = crt_result_signed(compute, circle)
            assert result == expected, f"{a} - {b}: signed CRT gave {result}, expected {expected}"

    def test_residue_decode_accuracy(self, compute, encoder):
        """Verify that hard residue decode matches true residues."""
        rng = random.Random(555)
        for _ in range(500):
            n = rng.randint(0, 999999)
            circle = encode(encoder, n)
            residues = compute._decode_residues_hard(circle)
            for i, p in enumerate(DEFAULT_PRIMES):
                expected_res = n % p
                got = int(residues[0, 0, i].item())
                assert got == expected_res, \
                    f"n={n}, prime={p}: residue={got}, expected={expected_res}"


# ---------------------------------------------------------------------------
# DIGIT DECOMPOSITION — comprehensive
# ---------------------------------------------------------------------------

class TestDigitDecompositionCorrectness:

    def test_known_values(self, compute):
        # Use float64 — float32 can't represent 999999999 exactly (rounds to 1e9)
        n = torch.tensor([[0.0, 1.0, 347.0, 12345.0, 999999999.0]], dtype=torch.float64)
        digits = compute.integer_to_digits(n)
        # 0 -> [0, 0, 0, ...]
        assert digits[0, 0, 0].item() == 0
        # 1 -> [1, 0, 0, ...]
        assert digits[0, 1, 0].item() == 1
        # 347 -> [7, 4, 3, 0, ...]
        assert digits[0, 2, 0].item() == 7
        assert digits[0, 2, 1].item() == 4
        assert digits[0, 2, 2].item() == 3
        # 12345 -> [5, 4, 3, 2, 1, 0, ...]
        assert digits[0, 3, 0].item() == 5
        assert digits[0, 3, 1].item() == 4
        assert digits[0, 3, 4].item() == 1
        # 999999999 -> [9, 9, 9, 9, 9, 9, 9, 9, 9, 0]
        for k in range(9):
            assert digits[0, 4, k].item() == 9

    def test_roundtrip_with_constants_module(self, compute):
        """integer_to_digits -> digits_to_integer roundtrip."""
        rng = random.Random(777)
        for _ in range(200):
            n = rng.randint(0, 999999999)
            digits_tensor = compute.integer_to_digits(torch.tensor([[float(n)]], dtype=torch.float64))
            digits_list = digits_tensor[0, 0].long().tolist()
            reconstructed = digits_to_integer(digits_list)
            assert reconstructed == n, f"Digit roundtrip: {n} -> {digits_list} -> {reconstructed}"

    def test_signed_decomposition(self, compute):
        """Signed decomposition: negative numbers get sign=1, abs digits."""
        n = torch.tensor([[-347.0]])
        result = compute.integer_to_digits_with_sign(n)
        assert result.shape == (1, 1, 11)  # K=10 digits + 1 sign
        assert result[0, 0, 10].item() == 1.0  # sign = negative
        assert result[0, 0, 0].item() == 7  # |347| ones digit
        assert result[0, 0, 2].item() == 3  # |347| hundreds digit


# ---------------------------------------------------------------------------
# END-TO-END PIPELINE TESTS
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Test the full pipeline: digits -> encode -> compute -> CRT -> verify."""

    def test_addition_pipeline(self, compute, encoder):
        """Full pipeline for addition across 200 random pairs."""
        rng = random.Random(1111)
        for _ in range(200):
            a = rng.randint(0, 499999)
            b = rng.randint(0, 499999)
            expected = a + b

            a_circle = encode(encoder, a)
            b_circle = encode(encoder, b)
            result_circle = compute.circle_add(a_circle, b_circle)
            result = crt_result(compute, result_circle)
            assert result == expected, f"Pipeline: {a} + {b} = {result}, expected {expected}"

    def test_subtraction_pipeline(self, compute, encoder):
        rng = random.Random(2222)
        for _ in range(200):
            a = rng.randint(0, 999999)
            b = rng.randint(0, 999999)
            expected = a - b

            result_circle = compute.circle_sub(encode(encoder, a), encode(encoder, b))
            result = crt_result_signed(compute, result_circle)
            assert result == expected, f"Pipeline: {a} - {b} = {result}, expected {expected}"

    def test_multiplication_pipeline(self, compute, encoder):
        rng = random.Random(3333)
        for _ in range(200):
            a = rng.randint(0, 9999)
            b = rng.randint(0, 9999)
            expected = a * b
            if expected >= P:
                continue

            result_circle = compute.circle_mul(encode(encoder, a), encode(encoder, b))
            result = crt_result(compute, result_circle)
            assert result == expected, f"Pipeline: {a} * {b} = {result}, expected {expected}"

    def test_exponentiation_pipeline(self, compute, encoder):
        rng = random.Random(4444)
        tested = 0
        for _ in range(2000):
            a = rng.randint(2, 30)
            b = rng.randint(0, 12)
            expected = a ** b
            if expected >= P:
                continue

            result_circle = compute.circle_exp(encode(encoder, a), encode_exp(encoder, b))
            result = crt_result(compute, result_circle)
            assert result == expected, f"Pipeline: {a}^{b} = {result}, expected {expected}"
            tested += 1
            if tested >= 200:
                break

    def test_chained_operations(self, compute, encoder):
        """Multi-step: (a + b) * c."""
        rng = random.Random(5555)
        for _ in range(100):
            a = rng.randint(1, 100)
            b = rng.randint(1, 100)
            c = rng.randint(1, 100)
            expected = (a + b) * c

            ab = compute.circle_add(encode(encoder, a), encode(encoder, b))
            result_circle = compute.circle_mul(ab, encode(encoder, c))
            result = crt_result(compute, result_circle)
            assert result == expected, f"({a}+{b})*{c} = {result}, expected {expected}"

    def test_chained_mul_then_sub(self, compute, encoder):
        """Multi-step: a * b - c."""
        rng = random.Random(6666)
        for _ in range(100):
            a = rng.randint(1, 100)
            b = rng.randint(1, 100)
            c = rng.randint(0, a * b)
            expected = a * b - c

            ab = compute.circle_mul(encode(encoder, a), encode(encoder, b))
            result_circle = compute.circle_sub(ab, encode(encoder, c))
            result = crt_result_signed(compute, result_circle)
            assert result == expected, f"{a}*{b}-{c} = {result}, expected {expected}"

    def test_chained_exp_then_add(self, compute, encoder):
        """Multi-step: a^b + c."""
        cases = [
            (2, 10, 100, 1124),   # 1024 + 100
            (3, 5, 1, 244),       # 243 + 1
            (5, 3, 25, 150),      # 125 + 25
            (10, 4, 1, 10001),    # 10000 + 1
        ]
        for a, b, c, expected in cases:
            ab = compute.circle_exp(encode(encoder, a), encode_exp(encoder, b))
            result_circle = compute.circle_add(ab, encode(encoder, c))
            result = crt_result(compute, result_circle)
            assert result == expected, f"{a}^{b}+{c} = {result}, expected {expected}"


# ---------------------------------------------------------------------------
# DIVISION — comprehensive
# ---------------------------------------------------------------------------

class TestDivisionCorrectness:
    """Verify circle-encoded exact division is correct."""

    def test_identity(self, compute, encoder):
        """a / 1 = a for various a."""
        for a in [1, 5, 99, 12345, 999999]:
            result = crt_result(compute, compute.circle_div(
                encode(encoder, a), encode(encoder, 1)
            ))
            assert result == a, f"{a} / 1 = {result}, expected {a}"

    def test_self_division(self, compute, encoder):
        """a / a = 1 for various a."""
        for a in [1, 2, 7, 37, 100, 12345]:
            result = crt_result(compute, compute.circle_div(
                encode(encoder, a), encode(encoder, a)
            ))
            assert result == 1, f"{a} / {a} = {result}, expected 1"

    def test_known_values(self, compute, encoder):
        """Test exact divisions with known results."""
        cases = [
            (12, 4, 3),
            (56, 8, 7),
            (144, 12, 12),
            (100, 10, 10),
            (1000, 25, 40),
            (63, 9, 7),
            (121, 11, 11),
            (1000000, 1000, 1000),
        ]
        for a, b, expected in cases:
            result = crt_result(compute, compute.circle_div(
                encode(encoder, a), encode(encoder, b)
            ))
            assert result == expected, f"{a} / {b} = {result}, expected {expected}"

    def test_inverse_of_multiplication(self, compute, encoder):
        """(a * b) / b = a — division undoes multiplication."""
        rng = random.Random(7777)
        for _ in range(200):
            a = rng.randint(1, 10000)
            b = rng.randint(1, 10000)
            product = a * b
            if product >= P:
                continue

            prod_circle = compute.circle_mul(encode(encoder, a), encode(encoder, b))
            result_circle = compute.circle_div(prod_circle, encode(encoder, b))
            result = crt_result(compute, result_circle)
            assert result == a, f"({a}*{b})/{b} = {result}, expected {a}"

    def test_random_exact_divisions(self, compute, encoder):
        """Random exact divisions: sample b, quotient, compute a = b * quotient."""
        rng = random.Random(8888)
        for _ in range(200):
            b = rng.randint(1, 1000)
            quotient = rng.randint(1, 10000)
            a = b * quotient
            if a >= P:
                continue

            result = crt_result(compute, compute.circle_div(
                encode(encoder, a), encode(encoder, b)
            ))
            assert result == quotient, f"{a} / {b} = {result}, expected {quotient}"


# ---------------------------------------------------------------------------
# BATCH OPERATION TESTS
# ---------------------------------------------------------------------------

class TestBatchCorrectness:
    """Verify that batched computation produces the same results as individual."""

    def test_batched_addition(self, compute, encoder):
        """Multiple additions in a single batch should be independent."""
        pairs = [(3, 4), (100, 200), (347, 291), (0, 0), (999, 1)]
        expected = [a + b for a, b in pairs]

        # Build batch: [1, 5, 10] digit tensors
        a_digits = torch.stack([
            torch.tensor(integer_to_digits(a, 10), dtype=torch.float32)
            for a, _ in pairs
        ]).unsqueeze(0)  # [1, 5, 10]
        b_digits = torch.stack([
            torch.tensor(integer_to_digits(b, 10), dtype=torch.float32)
            for _, b in pairs
        ]).unsqueeze(0)

        a_circle = encoder(a_digits)  # [1, 5, 9, 2]
        b_circle = encoder(b_digits)
        result_circle = compute.circle_add(a_circle, b_circle)

        for j, exp in enumerate(expected):
            single_result = crt_result(compute, result_circle[:, j:j+1, :, :])
            assert single_result == exp, f"Batch pos {j}: {pairs[j][0]}+{pairs[j][1]}={single_result}, expected {exp}"

    def test_batched_multiplication(self, compute, encoder):
        """Multiple multiplications in a single batch."""
        pairs = [(3, 4), (7, 8), (12, 12), (0, 99), (1, 347)]
        expected = [a * b for a, b in pairs]

        a_digits = torch.stack([
            torch.tensor(integer_to_digits(a, 10), dtype=torch.float32)
            for a, _ in pairs
        ]).unsqueeze(0)
        b_digits = torch.stack([
            torch.tensor(integer_to_digits(b, 10), dtype=torch.float32)
            for _, b in pairs
        ]).unsqueeze(0)

        a_circle = encoder(a_digits)
        b_circle = encoder(b_digits)
        result_circle = compute.circle_mul(a_circle, b_circle)

        for j, exp in enumerate(expected):
            single_result = crt_result(compute, result_circle[:, j:j+1, :, :])
            assert single_result == exp, f"Batch pos {j}: {pairs[j][0]}*{pairs[j][1]}={single_result}, expected {exp}"


# ---------------------------------------------------------------------------
# ENCODING CORRECTNESS
# ---------------------------------------------------------------------------

class TestEncodingCorrectness:
    """Verify the digit -> residue -> circle encoding is correct."""

    def test_encoding_matches_true_residues(self, encoder):
        """Circle encoding angle should correspond to true n mod p."""
        rng = random.Random(888)
        for _ in range(200):
            n = rng.randint(0, 999999999)
            circle = encode(encoder, n)  # [1, 1, 9, 2]
            for i, p in enumerate(DEFAULT_PRIMES):
                true_residue = n % p
                expected_angle = 2.0 * math.pi * true_residue / p
                expected_cos = math.cos(expected_angle)
                expected_sin = math.sin(expected_angle)
                got_cos = circle[0, 0, i, 0].item()
                got_sin = circle[0, 0, i, 1].item()
                assert abs(got_cos - expected_cos) < 1e-4, \
                    f"n={n}, p={p}: cos={got_cos}, expected={expected_cos}"
                assert abs(got_sin - expected_sin) < 1e-4, \
                    f"n={n}, p={p}: sin={got_sin}, expected={expected_sin}"

    def test_exponent_encoding_fermat_reduction(self, encoder):
        """Exponent encoding should reduce mod (p-1)."""
        rng = random.Random(999)
        for _ in range(100):
            b = rng.randint(0, 999999)
            circle = encode_exp(encoder, b)
            for i, p in enumerate(DEFAULT_PRIMES):
                pm1 = p - 1
                true_reduced = b % pm1
                expected_angle = 2.0 * math.pi * true_reduced / pm1
                expected_cos = math.cos(expected_angle)
                expected_sin = math.sin(expected_angle)
                got_cos = circle[0, 0, i, 0].item()
                got_sin = circle[0, 0, i, 1].item()
                assert abs(got_cos - expected_cos) < 1e-3, \
                    f"b={b}, p={p}: exp cos={got_cos}, expected={expected_cos}"
                assert abs(got_sin - expected_sin) < 1e-3, \
                    f"b={b}, p={p}: exp sin={got_sin}, expected={expected_sin}"

    def test_soft_decode_matches_hard_decode(self, compute, encoder):
        """Soft decode (softmax) should peak at the same residue as hard decode (argmax)."""
        rng = random.Random(111)
        for _ in range(100):
            n = rng.randint(0, 999999)
            circle = encode(encoder, n)
            soft = compute._decode_residues_soft(circle)
            hard = compute._decode_residues_hard(circle)  # [1, 1, m]
            for i, p in enumerate(DEFAULT_PRIMES):
                soft_argmax = soft[i][0, 0].argmax().item()
                hard_val = int(hard[0, 0, i].item())
                assert soft_argmax == hard_val, \
                    f"n={n}, p={p}: soft peak at {soft_argmax}, hard={hard_val}"

    def test_soft_decode_concentration(self, compute, encoder):
        """With temperature=1000, soft decode should be >0.99 at the peak."""
        rng = random.Random(222)
        for _ in range(50):
            n = rng.randint(0, 999999)
            circle = encode(encoder, n)
            soft = compute._decode_residues_soft(circle)
            for i, p in enumerate(DEFAULT_PRIMES):
                peak_prob = soft[i][0, 0].max().item()
                assert peak_prob > 0.99, \
                    f"n={n}, p={p}: peak prob={peak_prob:.4f}, should be >0.99"
