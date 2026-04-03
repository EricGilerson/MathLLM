"""Device-compatibility regressions for ARB modules."""

import torch

from mathllm.arb.constants import DEFAULT_PRIMES
from mathllm.arb.stage3_compute import ArithmeticCompute


def test_arithmetic_compute_has_no_float64_buffers():
    """Registered buffers must remain MPS-compatible."""
    compute = ArithmeticCompute(DEFAULT_PRIMES, num_digits=10, softmax_temperature=1000.0)

    float64_buffers = [
        name for name, buffer in compute.named_buffers() if buffer.dtype == torch.float64
    ]

    assert float64_buffers == []
