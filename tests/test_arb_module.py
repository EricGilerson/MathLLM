"""Tests for the complete Arithmetic Residual Block."""

import torch

from mathllm.arb.arb_module import ArithmeticResidualBlock
from mathllm.arb.constants import DEFAULT_PRIMES


def _build_test_arb(hidden_dim=64, **kwargs):
    """Create an ARB with a digit lookup table for testing."""
    arb = ArithmeticResidualBlock(hidden_dim=hidden_dim, primes=DEFAULT_PRIMES, **kwargs)
    # Build a simple digit table: token IDs 0-99 map to their digit values
    table = torch.zeros(100, 10)
    for i in range(100):
        v = i
        for d in range(10):
            table[i, d] = v % 10
            v //= 10
    arb.extract.token_digits = table
    # Set up operator detection: token 10 = '+', token 12 = '-'
    op_ids = torch.zeros(100, dtype=torch.bool)
    op_ids[10] = True  # +
    op_ids[12] = True  # -
    arb.extract.is_operator = op_ids
    return arb


class TestARBModule:
    def test_output_shape(self):
        arb = _build_test_arb()
        h = torch.randn(2, 4, 64)
        # "25 + 50 =" -> tokens [25, 10(+), 50, 99(=)]
        ids = torch.tensor([[25, 10, 50, 99], [30, 12, 40, 99]])
        h_out, d_a, d_b = arb(h, ids)
        assert h_out.shape == (2, 4, 64)
        assert d_a.shape == (2, 4, 10)
        assert d_b.shape == (2, 4, 10)

    def test_deterministic_extraction(self):
        """Extraction should return exact digit vectors from token IDs."""
        arb = _build_test_arb()
        h = torch.randn(1, 4, 64)
        # "25 + 50 =" -> A=25, B=50
        ids = torch.tensor([[25, 10, 50, 99]])
        _, d_a, d_b = arb(h, ids)
        # d_a should be digits of 25: [5, 2, 0, ...] (LSB first)
        assert d_a[0, 0, 0].item() == 5  # ones digit
        assert d_a[0, 0, 1].item() == 2  # tens digit
        # d_b should be digits of 50: [0, 5, 0, ...]
        assert d_b[0, 0, 0].item() == 0
        assert d_b[0, 0, 1].item() == 5

    def test_zero_init_is_identity(self):
        """With zero-initialized W_proj and zero gate, ARB should be a no-op."""
        arb = _build_test_arb(
            injector_init_std=0.0,
            gate_init_logit=-100.0,
        )
        h = torch.randn(2, 4, 64)
        ids = torch.tensor([[25, 10, 50, 99], [30, 12, 40, 99]])
        h_out, _, _ = arb(h, ids)
        assert torch.allclose(h_out, h, atol=1e-6), \
            "Zero-init ARB should produce h' = h"

    def test_default_injector_init_is_non_zero(self):
        arb = _build_test_arb()
        assert arb.inject.projection.weight.abs().sum() > 0

    def test_gradient_flow_to_injection(self):
        """Gradients should flow to the injection weights."""
        arb = _build_test_arb()
        with torch.no_grad():
            arb.inject.projection.weight.fill_(0.01)

        h = torch.randn(2, 4, 64, requires_grad=True)
        ids = torch.tensor([[25, 10, 50, 99], [30, 12, 40, 99]])
        h_out, _, _ = arb(h, ids)
        loss = h_out.sum()
        loss.backward()

        assert arb.inject.projection.weight.grad is not None
        assert arb.inject.projection.weight.grad.abs().sum() > 0

    def test_frozen_stages_no_gradients(self):
        arb = _build_test_arb()
        for param in arb.encode.parameters():
            assert not param.requires_grad
        for param in arb.compute.parameters():
            assert not param.requires_grad
        # Injection should be trainable
        for param in arb.inject.parameters():
            assert param.requires_grad

    def test_batch_independence(self):
        """Different batch elements should produce independent results."""
        arb = _build_test_arb()
        arb.eval()
        with torch.no_grad():
            arb.inject.projection.weight.fill_(0.01)

        h1 = torch.randn(1, 4, 64)
        h2 = torch.randn(1, 4, 64)
        h_batch = torch.cat([h1, h2], dim=0)
        ids1 = torch.tensor([[25, 10, 50, 99]])
        ids2 = torch.tensor([[30, 12, 40, 99]])
        ids_batch = torch.cat([ids1, ids2], dim=0)

        out_batch, _, _ = arb(h_batch, ids_batch)
        out1, _, _ = arb(h1, ids1)
        out2, _, _ = arb(h2, ids2)

        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)
