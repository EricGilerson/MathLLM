"""Tests for the complete Arithmetic Residual Block."""

import torch

from mathllm.arb.arb_module import ArithmeticResidualBlock
from mathllm.arb.constants import DEFAULT_PRIMES


def _make_dummy_ids(batch, seq_len, vocab_size=50257):
    """Create random token IDs for testing."""
    return torch.randint(0, vocab_size, (batch, seq_len))


class TestARBModule:
    def test_output_shape(self):
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        h = torch.randn(2, 5, 64)
        ids = _make_dummy_ids(2, 5)
        h_out, d_a, d_b, _, _ = arb(h, ids)
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
        ids = _make_dummy_ids(2, 5)
        h_out, _, _, _, _ = arb(h, ids)
        assert torch.allclose(h_out, h, atol=1e-6), \
            "Zero-init ARB should produce h' = h"

    def test_default_injector_init_is_non_zero(self):
        """Default init should be small but non-zero so gradients can propagate."""
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        assert arb.inject.projection.weight.abs().sum() > 0

    def test_gradient_flow_to_extraction(self):
        """Gradients should flow back to the extraction attention weights."""
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        # Build a simple digit table so token IDs 0-9 map to their digit values
        table = torch.zeros(100, 10)
        for i in range(100):
            v = i
            for d in range(10):
                table[i, d] = v % 10
                v //= 10
        arb.extract.token_digits = table
        arb.extract._table_built = True

        # Set W_proj to non-zero so gradients can flow
        with torch.no_grad():
            arb.inject.projection.weight.fill_(0.01)

        h = torch.randn(2, 3, 64, requires_grad=True)
        # Use number token IDs so digit lookup returns non-zero values
        ids = torch.tensor([[10, 25, 50], [30, 45, 99]])
        h_out, _, _, _, _ = arb(h, ids)
        loss = h_out.sum()
        loss.backward()

        # Check attention Q/K weights received gradients
        assert arb.extract.q_proj_a.weight.grad is not None
        assert arb.extract.q_proj_a.weight.grad.abs().sum() > 0

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

    def test_batch_independence(self):
        """Different batch elements should produce independent results."""
        arb = ArithmeticResidualBlock(hidden_dim=64, primes=DEFAULT_PRIMES)
        arb.eval()  # Disable dropout for deterministic batch independence check
        with torch.no_grad():
            arb.inject.projection.weight.fill_(0.01)

        h1 = torch.randn(1, 3, 64)
        h2 = torch.randn(1, 3, 64)
        h_batch = torch.cat([h1, h2], dim=0)
        ids1 = _make_dummy_ids(1, 3)
        ids2 = _make_dummy_ids(1, 3)
        ids_batch = torch.cat([ids1, ids2], dim=0)

        out_batch, _, _, _, _ = arb(h_batch, ids_batch)
        out1, _, _, _, _ = arb(h1, ids1)
        out2, _, _, _, _ = arb(h2, ids2)

        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)
