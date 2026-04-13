"""Auxiliary losses for ARB training.

With deterministic extraction, the aux loss is a no-op — extraction is exact
by construction. This module is kept for API compatibility with the trainer.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_extraction_loss(
    arb_extractions: dict[int, tuple[Tensor, Tensor, Tensor]],
    gt_digits_a: Tensor,
    gt_digits_b: Tensor,
    has_aux: Tensor,
    attention_mask: Tensor,
    eq_positions: Tensor,
    op_positions: Tensor | None = None,
) -> Tensor:
    """Return zero loss — extraction is deterministic, no training needed.

    API signature is preserved for compatibility with the trainer.
    """
    return gt_digits_a.new_zeros((), requires_grad=True)
