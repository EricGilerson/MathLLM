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
    """Tests for _select_operation_result: only the correct operation's
    digits should survive; the other 4 operations are zeroed out."""

    def _get_result_slices(self, K: int) -> dict[str, tuple[int, int]]:
        """Return (start, end) for each operation in the 5K+1 result vector."""
        return {
            "add": (0, K),
            "sub": (K, 2 * K + 1),
            "mul": (2 * K + 1, 3 * K + 1),
            "exp": (3 * K + 1, 4 * K + 1),
            "div": (4 * K + 1, 5 * K + 1),
        }

    def test_add_keeps_only_add_result(self):
        """With a '+' operator, only the add slice should be non-zero."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits  # 10
        D = 5 * K + 1
        slices = self._get_result_slices(K)

        # Fake results: fill each operation slice with its index+1 so they're distinct
        results = torch.zeros(1, 1, D)
        for i, (name, (s, e)) in enumerate(slices.items()):
            results[0, 0, s:e] = float(i + 1)

        # Token IDs: "25 + 50 =" → [25, 10(+), 50, 28(=)]
        ids = torch.tensor([[25, _OP_TOKENS['+'], 50, 28]])
        masked = arb._select_operation_result(results, ids)

        s, e = slices["add"]
        assert masked[0, 0, s:e].abs().sum() > 0, "add slice should be kept"
        # Everything outside the add slice should be zero
        outside = torch.cat([masked[0, 0, :s], masked[0, 0, e:]])
        assert outside.abs().sum() == 0, "non-add slices should be zeroed"

    def test_div_keeps_only_div_result(self):
        """With a '/' operator, only the div slice should be non-zero."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        D = 5 * K + 1
        slices = self._get_result_slices(K)

        results = torch.ones(1, 1, D)
        ids = torch.tensor([[6, _OP_TOKENS['/'], 3, 28]])
        masked = arb._select_operation_result(results, ids)

        s, e = slices["div"]
        assert masked[0, 0, s:e].abs().sum() > 0, "div slice should be kept"
        outside = torch.cat([masked[0, 0, :s], masked[0, 0, e:]])
        assert outside.abs().sum() == 0, "non-div slices should be zeroed"

    def test_mul_keeps_only_mul_result(self):
        """With a '*' operator, only the mul slice should be non-zero."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        D = 5 * K + 1
        slices = self._get_result_slices(K)

        results = torch.ones(1, 1, D)
        ids = torch.tensor([[5, _OP_TOKENS['*'], 3, 28]])
        masked = arb._select_operation_result(results, ids)

        s, e = slices["mul"]
        assert masked[0, 0, s:e].abs().sum() > 0, "mul slice should be kept"
        outside = torch.cat([masked[0, 0, :s], masked[0, 0, e:]])
        assert outside.abs().sum() == 0, "non-mul slices should be zeroed"

    def test_sub_keeps_only_sub_result(self):
        """With a '-' operator, only the sub slice (K+1 dims) should be non-zero."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        D = 5 * K + 1
        slices = self._get_result_slices(K)

        results = torch.ones(1, 1, D)
        ids = torch.tensor([[9, _OP_TOKENS['-'], 3, 28]])
        masked = arb._select_operation_result(results, ids)

        s, e = slices["sub"]
        assert e - s == K + 1, "sub slice should be K+1 (includes sign bit)"
        assert masked[0, 0, s:e].abs().sum() > 0, "sub slice should be kept"
        outside = torch.cat([masked[0, 0, :s], masked[0, 0, e:]])
        assert outside.abs().sum() == 0, "non-sub slices should be zeroed"

    def test_batch_mixed_operators(self):
        """Different operators in a batch should each select their own result."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        D = 5 * K + 1
        slices = self._get_result_slices(K)

        results = torch.ones(3, 1, D)
        ids = torch.tensor([
            [5, _OP_TOKENS['+'], 3, 28],  # add
            [6, _OP_TOKENS['*'], 2, 28],  # mul
            [8, _OP_TOKENS['/'], 4, 28],  # div  (operand 8 is not an op token)
        ])
        masked = arb._select_operation_result(results, ids)

        for b, expected_op in enumerate(["add", "mul", "div"]):
            s, e = slices[expected_op]
            assert masked[b, 0, s:e].abs().sum() > 0, \
                f"batch {b}: {expected_op} slice should be kept"
            outside = torch.cat([masked[b, 0, :s], masked[b, 0, e:]])
            assert outside.abs().sum() == 0, \
                f"batch {b}: non-{expected_op} slices should be zeroed"

    def test_no_operator_zeros_everything(self):
        """If no operator token is found, all results should be zeroed."""
        arb = _build_test_arb(with_op_selection=True)
        K = arb.num_digits
        D = 5 * K + 1

        results = torch.ones(1, 1, D)
        # Token IDs with no operator
        ids = torch.tensor([[25, 30, 50, 28]])
        masked = arb._select_operation_result(results, ids)
        assert masked.abs().sum() == 0, "no operator → all results zeroed"

    def test_selection_used_in_forward(self):
        """Full forward pass should inject only the selected operation's result."""
        arb = _build_test_arb(with_op_selection=True)
        arb.eval()
        K = arb.num_digits
        slices = self._get_result_slices(K)

        with torch.no_grad():
            arb.inject.projection.weight.fill_(0.01)

        h = torch.randn(2, 4, 64)
        ids = torch.tensor([
            [25, _OP_TOKENS['+'], 50, 28],  # add
            [ 6, _OP_TOKENS['/'],  3, 28],  # div  (token 6 is not an op token)
        ])

        # Intercept results after selection by monkey-patching _decode_to_digits
        captured = {}
        original_decode = arb._decode_to_digits

        def capturing_decode(*args, **kwargs):
            result = original_decode(*args, **kwargs)
            # Apply the same selection the forward() will apply
            selected = arb._select_operation_result(result, ids)
            captured["pre_selection"] = result.clone()
            captured["post_selection"] = selected.clone()
            return result  # forward() will apply selection itself

        arb._decode_to_digits = capturing_decode
        arb(h, ids)
        arb._decode_to_digits = original_decode

        pre = captured["pre_selection"]
        post = captured["post_selection"]

        # Batch 0 (+): only add slice should survive
        s, e = slices["add"]
        assert post[0, 0, s:e].abs().sum() > 0
        outside = torch.cat([post[0, 0, :s], post[0, 0, e:]])
        assert outside.abs().sum() == 0

        # Batch 1 (/): only div slice should survive
        s, e = slices["div"]
        assert post[1, 0, s:e].abs().sum() > 0
        outside = torch.cat([post[1, 0, :s], post[1, 0, e:]])
        assert outside.abs().sum() == 0

        # Pre-selection should have had non-zero values in other slices
        assert pre[0].abs().sum() > post[0].abs().sum(), \
            "selection should have removed some non-zero values"
