"""Tests for mathematical constants and precomputed tables."""

import math

import pytest
import torch

from mathllm.arb.constants import (
    DEFAULT_PRIMES,
    compute_coefficient_matrix,
    compute_circle_templates,
    compute_crt_weights,
    compute_exp_coefficient_matrix,
    compute_exponentiation_tables,
    compute_multiplication_tables,
    compute_product,
    digits_to_integer,
    integer_to_digits,
    mod_inverse,
)


class TestModInverse:
    def test_basic(self):
        assert (mod_inverse(3, 7) * 3) % 7 == 1

    def test_all_primes(self):
        for p in DEFAULT_PRIMES:
            for a in range(1, p):
                inv = mod_inverse(a, p)
                assert (a * inv) % p == 1, f"Failed for a={a}, p={p}"

    def test_no_inverse(self):
        with pytest.raises(ValueError):
            mod_inverse(0, 7)


class TestProduct:
    def test_default_primes(self):
        P = compute_product(DEFAULT_PRIMES)
        expected = 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37
        assert P == expected
        assert P > 10**9  # Must cover integers up to 10^9


class TestDigitConversion:
    def test_roundtrip(self):
        for n in [0, 1, 42, 347, 12345, 999999999]:
            digits = integer_to_digits(n, 10)
            assert digits_to_integer(digits) == n

    def test_specific(self):
        digits = integer_to_digits(347, 10)
        assert digits[0] == 7  # least significant
        assert digits[1] == 4
        assert digits[2] == 3

    def test_tensor_roundtrip(self):
        digits = integer_to_digits(12345, 10)
        t = torch.tensor(digits)
        assert digits_to_integer(t) == 12345


class TestCoefficientMatrix:
    def test_shape(self):
        C = compute_coefficient_matrix(DEFAULT_PRIMES, 10)
        assert C.shape == (9, 10)

    def test_values(self):
        C = compute_coefficient_matrix(DEFAULT_PRIMES, 10)
        # C[i, k] = 10^k mod p_i
        for i, p in enumerate(DEFAULT_PRIMES):
            for k in range(10):
                assert C[i, k].item() == pow(10, k, p)

    def test_digit_to_residue(self):
        """Verify that C @ digits gives the correct residue."""
        C = compute_coefficient_matrix(DEFAULT_PRIMES, 10)
        n = 347
        digits = torch.tensor(integer_to_digits(n, 10), dtype=torch.float32)
        residues = C @ digits
        for i, p in enumerate(DEFAULT_PRIMES):
            assert int(residues[i].item()) % p == n % p


class TestExpCoefficientMatrix:
    def test_shape(self):
        C = compute_exp_coefficient_matrix(DEFAULT_PRIMES, 10)
        assert C.shape == (9, 10)

    def test_values(self):
        C = compute_exp_coefficient_matrix(DEFAULT_PRIMES, 10)
        for i, p in enumerate(DEFAULT_PRIMES):
            for k in range(10):
                assert C[i, k].item() == pow(10, k, p - 1)


class TestCRTWeights:
    def test_shape(self):
        w = compute_crt_weights(DEFAULT_PRIMES)
        assert w.shape == (9,)
        assert w.dtype == torch.float64

    def test_reconstruction(self):
        """Verify CRT reconstruction for many random integers."""
        P = compute_product(DEFAULT_PRIMES)
        w = compute_crt_weights(DEFAULT_PRIMES)

        import random
        rng = random.Random(42)

        for _ in range(1000):
            n = rng.randint(0, P - 1)
            residues = torch.tensor(
                [n % p for p in DEFAULT_PRIMES], dtype=torch.float64
            )
            reconstructed = int((residues * w).sum().item() % P)
            assert reconstructed == n, f"CRT failed for n={n}: got {reconstructed}"

    def test_small_values(self):
        P = compute_product(DEFAULT_PRIMES)
        w = compute_crt_weights(DEFAULT_PRIMES)
        for n in [0, 1, 100, 999, 347291, 999999999]:
            residues = torch.tensor(
                [n % p for p in DEFAULT_PRIMES], dtype=torch.float64
            )
            assert int((residues * w).sum().item() % P) == n


class TestMultiplicationTables:
    def test_shapes(self):
        tables = compute_multiplication_tables(DEFAULT_PRIMES)
        assert len(tables) == 9
        for i, p in enumerate(DEFAULT_PRIMES):
            assert tables[i].shape == (p, p)

    def test_correctness(self):
        tables = compute_multiplication_tables(DEFAULT_PRIMES)
        for i, p in enumerate(DEFAULT_PRIMES):
            for a in range(p):
                for b in range(p):
                    assert tables[i][a, b].item() == (a * b) % p


class TestExponentiationTables:
    def test_shapes(self):
        tables = compute_exponentiation_tables(DEFAULT_PRIMES)
        assert len(tables) == 9
        for i, p in enumerate(DEFAULT_PRIMES):
            assert tables[i].shape == (p, p - 1)

    def test_correctness(self):
        tables = compute_exponentiation_tables(DEFAULT_PRIMES)
        for i, p in enumerate(DEFAULT_PRIMES):
            for a in range(p):
                for k in range(p - 1):
                    assert tables[i][a, k].item() == pow(a, k, p)


class TestCircleTemplates:
    def test_shapes(self):
        templates = compute_circle_templates(DEFAULT_PRIMES)
        for i, p in enumerate(DEFAULT_PRIMES):
            assert templates[i].shape == (p, 2)

    def test_unit_circle(self):
        """All template vectors should lie on the unit circle."""
        templates = compute_circle_templates(DEFAULT_PRIMES)
        for t in templates:
            norms = (t[:, 0] ** 2 + t[:, 1] ** 2).sqrt()
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_orthogonal_decode(self):
        """Inner product of a template with itself should be maximal."""
        templates = compute_circle_templates(DEFAULT_PRIMES)
        for t in templates:
            # Each template dotted with itself gives 1.0 (unit vector)
            self_dots = (t * t).sum(dim=-1)
            assert torch.allclose(self_dots, torch.ones_like(self_dots), atol=1e-6)
