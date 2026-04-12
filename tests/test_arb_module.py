"""Tests for the complete Arithmetic Residual Block."""

import torch

from mathllm.arb.arb_module import ArithmeticResidualBlock
from mathllm.arb.constants import DEFAULT_PRIMES

# Token IDs for operators in the test vocabulary (0-99)
_OP_TOKENS = {'+': 10, '-': 12, '*': 14, '^': 16, '/': 18}


def _build_test_arb(hidden_dim=64, with_op_selection=False, **kwargs):
    """Create an ARB with a digit lookup table for testing.

    Args:
        with_op_selection: If True, build the op_token_to_result_idx table
            so that _select_operation_result is active (not skipped).
    """
    arb = ArithmeticResidualBlock(hidden_dim=hidden_dim, primes=DEFAULT_PRIMES, **kwargs)
    # Build a simple digit table: token IDs 0-99 map to their digit values
    table = torch.zeros(100, 10)
    for i in range(100):
        v = i
        for d in range(10):
            table[i, d] = v % 10
            v //= 10
    arb.extract.register_buffer("token_digits_full", table, persistent=True)
    arb.extract._per_digit = False
    # Set up operator detection
    op_ids = torch.zeros(100, dtype=torch.bool)
    for tid in _OP_TOKENS.values():
        op_ids[tid] = True
    arb.extract.is_operator = op_ids

    if with_op_selection:
        # Build the operator-to-result-index table (matches _decode_to_digits order)
        op_result_idx = torch.full((100,), -1, dtype=torch.long)
        _OP_TO_INDEX = {'+': 0, '-': 1, '*': 2, '^': 3, '/': 4}
        for op_char, idx in _OP_TO_INDEX.items():
            op_result_idx[_OP_TOKENS[op_char]] = idx
        arb.extract.register_buffer(
            "op_token_to_result_idx", op_result_idx, persistent=True,
        )

    return arb


class TestARBModule:
    def test_output_shape(self):
        arb = _build_test_arb()
        h = torch.randn(2, 4, 64)
        # "25 + 50 =" -> tokens [25, 10(+), 50, 99(=)]
        ids = torch.tensor([[25, 10, 50, 99], [30, 12, 40, 28]])
        h_out, d_a, d_b = arb(h, ids)
        assert h_out.shape == (2, 4, 64)
        assert d_a.shape == (2, 4, 10)
        assert d_b.shape == (2, 4, 10)

    def test_deterministic_extraction(self):
        """Extraction should return exact digit vectors from token IDs."""
        arb = _build_test_arb()
        h = torch.randn(1, 4, 64)
        # "25 + 50 =" -> A=25, B=50
        ids = torch.tensor([[25, 10, 50, 28]])
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
        ids = torch.tensor([[25, 10, 50, 99], [30, 12, 40, 28]])
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
        # Use token 28 for '=' so injection mask is active
        ids = torch.tensor([[25, 10, 50, 28], [30, 12, 40, 28]])
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
        ids1 = torch.tensor([[25, 10, 50, 28]])
        ids2 = torch.tensor([[30, 12, 40, 28]])
        ids_batch = torch.cat([ids1, ids2], dim=0)

        out_batch, _, _ = arb(h_batch, ids_batch)
        out1, _, _ = arb(h1, ids1)
        out2, _, _ = arb(h2, ids2)

        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)


class TestOperationSelection:
    """Tests for _select_operation_result: extracts the correct operation's
    digits into fixed positions [0:K] with sign bit at [K]."""

    def _build_fake_results(self, K: int) -> torch.Tensor:
        """Build a [1, 1, 5K+1] tensor with distinct values per operation.

        add=[1,1,...], sub=[2,2,...,sign=0.5], mul=[3,3,...], exp=[4,4,...], div=[5,5,...]
        """
        D = 5 * K + 1
        results = torch.zeros(1, 1, D)
        results[0, 0, 0:K] = 1.0                    # add
        results[0, 0, K:2 * K] = 2.0                # sub digits
        results[0, 0, 2 * K] = 0.5                  # sub sign bit
        results[0, 0, 2 * K + 1:3 * K + 1] = 3.0   # mul
        results[0, 0, 3 * K + 1:4 * K + 1] = 4.0   # exp
        results[0, 0, 4 * K + 1:5 * K + 1] = 5.0   # div
        return results

    def test_output_shape_is_K_plus_1(self):
        """Output should be [B, S, K+1] regardless of operation."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        D = 5 * K + 1
        results = torch.ones(2, 3, D)
        ids = torch.tensor([
            [5, _OP_TOKENS['+'], 3, 28],
            [9, _OP_TOKENS['-'], 3, 28],
        ])
        selected = arb._select_operation_result(results, ids)
        assert selected.shape == (2, 3, K + 1)

    def test_add_selects_add_digits(self):
        """With '+', output should contain the add digits with sign=0."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        results = self._build_fake_results(K)
        ids = torch.tensor([[25, _OP_TOKENS['+'], 50, 28]])
        selected = arb._select_operation_result(results, ids)

        assert (selected[0, 0, :K] == 1.0).all(), "digits should be add values"
        assert selected[0, 0, K].item() == 0.0, "sign bit should be 0 for add"

    def test_sub_selects_sub_digits_with_sign(self):
        """With '-', output should contain sub digits AND sign bit."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        results = self._build_fake_results(K)
        ids = torch.tensor([[9, _OP_TOKENS['-'], 3, 28]])
        selected = arb._select_operation_result(results, ids)

        assert (selected[0, 0, :K] == 2.0).all(), "digits should be sub values"
        assert selected[0, 0, K].item() == 0.5, "sign bit should be sub's sign (0.5)"

    def test_mul_selects_mul_digits(self):
        """With '*', output should contain mul digits with sign=0."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        results = self._build_fake_results(K)
        ids = torch.tensor([[5, _OP_TOKENS['*'], 3, 28]])
        selected = arb._select_operation_result(results, ids)

        assert (selected[0, 0, :K] == 3.0).all(), "digits should be mul values"
        assert selected[0, 0, K].item() == 0.0, "sign bit should be 0 for mul"

    def test_div_selects_div_digits(self):
        """With '/', output should contain div digits with sign=0."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        results = self._build_fake_results(K)
        ids = torch.tensor([[6, _OP_TOKENS['/'], 3, 28]])
        selected = arb._select_operation_result(results, ids)

        assert (selected[0, 0, :K] == 5.0).all(), "digits should be div values"
        assert selected[0, 0, K].item() == 0.0, "sign bit should be 0 for div"

    def test_exp_selects_exp_digits(self):
        """With '^', output should contain exp digits with sign=0."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        results = self._build_fake_results(K)
        ids = torch.tensor([[2, _OP_TOKENS['^'], 7, 28]])
        selected = arb._select_operation_result(results, ids)

        assert (selected[0, 0, :K] == 4.0).all(), "digits should be exp values"
        assert selected[0, 0, K].item() == 0.0, "sign bit should be 0 for exp"

    def test_batch_mixed_operators(self):
        """Different operators in a batch should each select their own result."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        results = self._build_fake_results(K).expand(3, 1, -1).clone()
        ids = torch.tensor([
            [5, _OP_TOKENS['+'], 3, 28],  # add → digits=1
            [6, _OP_TOKENS['*'], 2, 28],  # mul → digits=3
            [8, _OP_TOKENS['/'], 4, 28],  # div → digits=5
        ])
        selected = arb._select_operation_result(results, ids)

        assert (selected[0, 0, :K] == 1.0).all(), "batch 0: add digits"
        assert (selected[1, 0, :K] == 3.0).all(), "batch 1: mul digits"
        assert (selected[2, 0, :K] == 5.0).all(), "batch 2: div digits"

    def test_no_operator_zeros_everything(self):
        """If no operator token is found, all results should be zeroed."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        results = self._build_fake_results(K)
        ids = torch.tensor([[25, 30, 50, 28]])
        selected = arb._select_operation_result(results, ids)
        assert selected.abs().sum() == 0, "no operator → all results zeroed"

    def test_selection_used_in_forward(self):
        """Full forward pass should produce K+1 dim results before injection."""
        arb = _build_test_arb(with_op_selection=True)
        arb.eval()
        K = arb.num_digits

        with torch.no_grad():
            arb.inject.projection.weight.fill_(0.01)

        h = torch.randn(2, 4, 64)
        ids = torch.tensor([
            [25, _OP_TOKENS['+'], 50, 28],  # add
            [ 6, _OP_TOKENS['/'],  3, 28],  # div
        ])

        # Intercept the selected results
        captured = {}
        original_select = arb._select_operation_result

        def capturing_select(results, input_ids):
            selected = original_select(results, input_ids)
            captured["all_results"] = results.clone()
            captured["selected"] = selected.clone()
            return selected

        arb._select_operation_result = capturing_select
        arb(h, ids)
        arb._select_operation_result = original_select

        assert captured["all_results"].shape[-1] == 5 * K + 1, \
            "pre-selection should be 5K+1"
        assert captured["selected"].shape[-1] == K + 1, \
            "post-selection should be K+1"
        # Add and div should have selected different digit values
        assert not torch.allclose(
            captured["selected"][0, 0, :K],
            captured["selected"][1, 0, :K],
        ), "different operations should produce different selected digits"
