"""Tests for the complete Arithmetic Residual Block."""

import torch

from mathllm.arb.arb_module import ArithmeticResidualBlock
from mathllm.arb.constants import DEFAULT_PRIMES


class TestARBModule:
    def test_output_shape(self):
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        h = torch.randn(2, 5, 64)
        h_out = arb(h)
        assert h_out.shape == (2, 5, 64)

    def test_zero_init_is_identity(self):
        """With zero-initialized W_proj, ARB should be a no-op (output = input)."""
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        h = torch.randn(2, 5, 64)
        h_out = arb(h)
        assert torch.allclose(h_out, h, atol=1e-6), \
            "Zero-init ARB should produce h' = h"

    def test_gradient_flow_to_extraction(self):
        """Gradients should flow back to the extraction weights."""
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        # Set W_proj to non-zero so gradients can flow
        with torch.no_grad():
            arb.inject.projection.weight.fill_(0.01)

        h = torch.randn(2, 3, 64, requires_grad=True)
        h_out = arb(h)
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
        # Injection: result_dim = 4 * 9 * 2 = 72; Linear(72, 768) = 72*768 + 768 = 56064
        expected_learned = 15380 + 56064
        assert counts["learned"] == expected_learned, \
            f"Expected {expected_learned} learned params, got {counts['learned']}"

    def test_batch_independence(self):
        """Different batch elements should produce independent results."""
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        with torch.no_grad():
            arb.inject.projection.weight.fill_(0.01)

        h1 = torch.randn(1, 3, 64)
        h2 = torch.randn(1, 3, 64)
        h_batch = torch.cat([h1, h2], dim=0)

        out_batch = arb(h_batch)
        out1 = arb(h1)
        out2 = arb(h2)

        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)
