"""Tests for Straight-Through Estimator."""

import torch

from mathllm.arb.ste import ste_clamp, ste_round, ste_round_clamp


class TestSTERound:
    def test_forward_rounds(self):
        x = torch.tensor([1.3, 2.7, -0.4, 4.5])
        result = ste_round(x)
        expected = torch.tensor([1.0, 3.0, 0.0, 4.0])
        assert torch.equal(result, expected)

    def test_gradient_passes_through(self):
        x = torch.tensor([1.3, 2.7, -0.4], requires_grad=True)
        y = ste_round(x)
        loss = y.sum()
        loss.backward()
        # Gradient should be 1.0 for each element (identity in backward)
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_integer_input_unchanged(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(ste_round(x), x)


class TestSTEClamp:
    def test_forward_clamps(self):
        x = torch.tensor([-1.0, 5.0, 15.0])
        result = ste_clamp(x, 0.0, 9.0)
        expected = torch.tensor([0.0, 5.0, 9.0])
        assert torch.equal(result, expected)

    def test_gradient_passes_through(self):
        x = torch.tensor([-1.0, 5.0, 15.0], requires_grad=True)
        y = ste_clamp(x, 0.0, 9.0)
        loss = y.sum()
        loss.backward()
        assert torch.allclose(x.grad, torch.ones_like(x))


class TestSTERoundClamp:
    def test_combined(self):
        x = torch.tensor([-0.3, 4.7, 11.2])
        result = ste_round_clamp(x, low=0, high=9)
        expected = torch.tensor([0.0, 5.0, 9.0])
        assert torch.equal(result, expected)

    def test_gradient_passes_through(self):
        x = torch.tensor([3.3, 7.8, -2.1], requires_grad=True)
        y = ste_round_clamp(x, low=0, high=9)
        loss = y.sum()
        loss.backward()
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_valid_digit_range(self):
        """All outputs should be integers in [0, 9]."""
        x = torch.randn(100) * 20  # wide range
        result = ste_round_clamp(x)
        assert (result >= 0).all()
        assert (result <= 9).all()
        assert torch.equal(result, result.round())
