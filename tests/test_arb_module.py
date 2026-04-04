"""Tests for the complete Arithmetic Residual Block."""

import torch

from mathllm.arb.arb_module import ArithmeticResidualBlock
from mathllm.arb.constants import DEFAULT_PRIMES


class TestARBModule:
    def test_output_shape(self):
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        h = torch.randn(2, 5, 64)
        h_out, d_a, d_b = arb(h)
        assert h_out.shape == (2, 5, 64)
        assert d_a.shape == (2, 5, 10)
        assert d_b.shape == (2, 5, 10)

    def test_zero_init_is_identity(self):
        """With zero-initialized W_proj and zero gate, ARB should be a no-op."""
        arb = ArithmeticResidualBlock(
            hidden_dim=64,
            primes=DEFAULT_PRIMES,
            injector_init_std=0.0,
            gate_init_logit=-100.0,  # sigmoid(-100) ~ 0
        )
        h = torch.randn(2, 5, 64)
        h_out, _, _ = arb(h)
        assert torch.allclose(h_out, h, atol=1e-6), \
            "Zero-init ARB should produce h' = h"

    def test_default_injector_init_is_non_zero(self):
        """Default init should be small but non-zero so gradients can propagate."""
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        assert arb.inject.projection.weight.abs().sum() > 0

    def test_gradient_flow_to_extraction(self):
        """Gradients should flow back to the extraction weights."""
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        # Set W_proj to non-zero so gradients can flow
        with torch.no_grad():
            arb.inject.projection.weight.fill_(0.01)

        h = torch.randn(2, 3, 64, requires_grad=True)
        h_out, _, _ = arb(h)
        loss = h_out.sum()
        loss.backward()

        # Check extraction weights received gradients
        assert arb.extract.W_a.weight.grad is not None
        assert arb.extract.W_a.weight.grad.abs().sum() > 0

    def test_frozen_stages_no_gradients(self):
        """Frozen stages should not have requires_grad=True."""
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)

        # Encoder params should be frozen
        for param in arb.encode.parameters():
            assert not param.requires_grad

        # Compute params should be frozen
        for param in arb.compute.parameters():
            assert not param.requires_grad

        # Extract and inject should be trainable
        for param in arb.extract.parameters():
            assert param.requires_grad
        for param in arb.inject.parameters():
            assert param.requires_grad

    def test_parameter_count(self):
        arb = ArithmeticResidualBlock(hidden_dim=768, primes=DEFAULT_PRIMES)
        counts = arb.count_parameters()
        assert counts["learned"] > 0
        # Extraction: 2 * (768 * 10 + 10) = 15380
        # Injection: result_dim = 5 * 9 * 2 = 90; Linear(90, 768) = 90*768 + 768 = 69888
        # Gate: 1 scalar parameter
        expected_learned = 15380 + 69888 + 1
        assert counts["learned"] == expected_learned, \
            f"Expected {expected_learned} learned params, got {counts['learned']}"

    def test_batch_independence(self):
        """Different batch elements should produce independent results."""
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        arb.eval()  # Disable dropout for deterministic batch independence check
        with torch.no_grad():
            arb.inject.projection.weight.fill_(0.01)

        h1 = torch.randn(1, 3, 64)
        h2 = torch.randn(1, 3, 64)
        h_batch = torch.cat([h1, h2], dim=0)

        out_batch, _, _ = arb(h_batch)
        out1, _, _ = arb(h1)
        out2, _, _ = arb(h2)

        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)
