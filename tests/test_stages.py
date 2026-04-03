"""Tests for individual ARB stages."""

import math

import pytest
import torch

from mathllm.arb.constants import (
    DEFAULT_PRIMES,
    compute_product,
    integer_to_digits,
)
from mathllm.arb.stage1_extract import OperandExtractor
from mathllm.arb.stage2_encode import RNSCircleEncoder
from mathllm.arb.stage3_compute import ArithmeticCompute


class TestOperandExtractor:
    def test_output_shape(self):
        ext = OperandExtractor(hidden_dim=64, num_digits=10)
        h = torch.randn(2, 5, 64)
        d_a, d_b = ext(h)
        assert d_a.shape == (2, 5, 10)
        assert d_b.shape == (2, 5, 10)

    def test_output_range(self):
        ext = OperandExtractor(hidden_dim=64, num_digits=10)
        h = torch.randn(2, 5, 64) * 10  # large inputs
        d_a, d_b = ext(h)
        assert (d_a >= 0).all() and (d_a <= 9).all()
        assert (d_b >= 0).all() and (d_b <= 9).all()

    def test_output_integers(self):
        ext = OperandExtractor(hidden_dim=64, num_digits=10)
        h = torch.randn(2, 5, 64)
        d_a, _ = ext(h)
        assert torch.equal(d_a, d_a.round())

    def test_gradient_flow(self):
        ext = OperandExtractor(hidden_dim=64, num_digits=10)
        h = torch.randn(2, 5, 64, requires_grad=True)
        d_a, d_b = ext(h)
        loss = (d_a.sum() + d_b.sum())
        loss.backward()
        assert h.grad is not None
        assert h.grad.abs().sum() > 0


class TestRNSCircleEncoder:
    def test_output_shape(self):
        enc = RNSCircleEncoder(DEFAULT_PRIMES, num_digits=10)
        digits = torch.zeros(2, 3, 10)
        circle = enc(digits)
        assert circle.shape == (2, 3, 9, 2)

    def test_unit_circle(self):
        """Encoded values should lie on the unit circle."""
        enc = RNSCircleEncoder(DEFAULT_PRIMES, num_digits=10)
        digits = torch.tensor(integer_to_digits(347, 10), dtype=torch.float32)
        digits = digits.unsqueeze(0).unsqueeze(0)  # [1, 1, 10]
        circle = enc(digits)
        norms = (circle[..., 0] ** 2 + circle[..., 1] ** 2).sqrt()
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_known_value(self):
        """Check encoding of 100 with prime 7: 100 mod 7 = 2."""
        enc = RNSCircleEncoder(DEFAULT_PRIMES, num_digits=10)
        digits = torch.tensor(integer_to_digits(100, 10), dtype=torch.float32)
        digits = digits.unsqueeze(0).unsqueeze(0)
        circle = enc(digits)

        # Prime 7 is index 0
        expected_angle = 2.0 * math.pi * 2 / 7  # 100 mod 7 = 2
        expected_cos = math.cos(expected_angle)
        expected_sin = math.sin(expected_angle)
        assert abs(circle[0, 0, 0, 0].item() - expected_cos) < 1e-5
        assert abs(circle[0, 0, 0, 1].item() - expected_sin) < 1e-5

    def test_periodicity(self):
        """n and n+p should produce the same circle encoding for prime p."""
        enc = RNSCircleEncoder(DEFAULT_PRIMES, num_digits=10)
        n = 100
        for i, p in enumerate(DEFAULT_PRIMES):
            d1 = torch.tensor(integer_to_digits(n, 10), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            d2 = torch.tensor(integer_to_digits(n + p, 10), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            c1 = enc(d1)[0, 0, i]
            c2 = enc(d2)[0, 0, i]
            assert torch.allclose(c1, c2, atol=1e-4), f"Periodicity failed for p={p}"

    def test_exponent_encoding_shape(self):
        enc = RNSCircleEncoder(DEFAULT_PRIMES, num_digits=10)
        digits = torch.zeros(2, 3, 10)
        circle = enc.encode_exponent(digits)
        assert circle.shape == (2, 3, 9, 2)


class TestArithmeticCompute:
    @pytest.fixture
    def compute(self):
        return ArithmeticCompute(DEFAULT_PRIMES, num_digits=10, softmax_temperature=1000.0)

    @pytest.fixture
    def encoder(self):
        return RNSCircleEncoder(DEFAULT_PRIMES, num_digits=10)

    def _encode_number(self, encoder, n):
        """Helper: encode integer to circle representation."""
        digits = torch.tensor(integer_to_digits(n, 10), dtype=torch.float32)
        digits = digits.unsqueeze(0).unsqueeze(0)  # [1, 1, 10]
        return encoder(digits)

    def test_addition(self, compute, encoder):
        """Test that circle addition gives correct CRT result."""
        P = compute_product(DEFAULT_PRIMES)
        test_cases = [(3, 4, 7), (100, 200, 300), (347, 291, 638), (0, 0, 0)]

        for a, b, expected in test_cases:
            a_circle = self._encode_number(encoder, a)
            b_circle = self._encode_number(encoder, b)
            result_circle = compute.circle_add(a_circle, b_circle)
            result = compute.crt_reconstruct(result_circle)
            assert int(result.item()) % P == expected % P, f"{a} + {b}: expected {expected}, got {int(result.item())}"

    def test_subtraction(self, compute, encoder):
        """Test modular subtraction."""
        P = compute_product(DEFAULT_PRIMES)
        test_cases = [(10, 3, 7), (638, 291, 347), (100, 100, 0)]

        for a, b, expected in test_cases:
            a_circle = self._encode_number(encoder, a)
            b_circle = self._encode_number(encoder, b)
            result_circle = compute.circle_sub(a_circle, b_circle)
            result = compute.crt_reconstruct_signed(result_circle)
            assert int(result.item()) == expected, f"{a} - {b}: expected {expected}, got {int(result.item())}"

    def test_subtraction_negative(self, compute, encoder):
        """Test subtraction with negative result."""
        a_circle = self._encode_number(encoder, 3)
        b_circle = self._encode_number(encoder, 10)
        result = compute.crt_reconstruct_signed(compute.circle_sub(a_circle, b_circle))
        assert int(result.item()) == -7

    def test_multiplication(self, compute, encoder):
        """Test multiplication via table lookup."""
        test_cases = [(3, 4, 12), (7, 8, 56), (12, 12, 144), (0, 99, 0), (1, 347, 347)]

        for a, b, expected in test_cases:
            a_circle = self._encode_number(encoder, a)
            b_circle = self._encode_number(encoder, b)
            result_circle = compute.circle_mul(a_circle, b_circle)
            result = compute.crt_reconstruct(result_circle)
            assert int(result.item()) == expected, f"{a} * {b}: expected {expected}, got {int(result.item())}"

    def test_digit_decomposition(self, compute):
        """Test integer to digit vector conversion."""
        n = torch.tensor([[347.0]])  # [1, 1]
        digits = compute.integer_to_digits(n)
        assert digits.shape == (1, 1, 10)
        assert digits[0, 0, 0].item() == 7  # ones
        assert digits[0, 0, 1].item() == 4  # tens
        assert digits[0, 0, 2].item() == 3  # hundreds

    def test_full_forward(self, compute, encoder):
        """Test the full forward pass produces flattened circle results."""
        a_circle = self._encode_number(encoder, 5)
        b_circle = self._encode_number(encoder, 3)
        b_exp_circle = encoder.encode_exponent(
            torch.tensor(integer_to_digits(3, 10), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

        results = compute(a_circle, b_circle, b_exp_circle)
        # 4 operations * 9 primes * 2 (cos, sin) = 72
        assert results.shape == (1, 1, 72)
