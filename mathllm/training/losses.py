"""Auxiliary losses for ARB training.

The extraction loss supervises the soft digit outputs (attention-weighted
digit vectors) against ground-truth digit values using MSE. Both heads are
supervised at the '=' position where the full expression is visible.
Separate Q/K attention projections per head learn to attend to different
operand token positions.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_extraction_loss(
    arb_extractions: dict[int, tuple[Tensor, Tensor]],
    gt_digits_a: Tensor,
    gt_digits_b: Tensor,
    has_aux: Tensor,
    attention_mask: Tensor,
    eq_positions: Tensor,
    op_positions: Tensor | None = None,
) -> Tensor:
    """Compute MSE loss between soft-extracted digits and ground-truth digits.

    Both heads are supervised at the '=' position. The attention mechanism
    differentiates which operand to attend to via separate Q/K projections.

    Args:
        arb_extractions: {layer_id: (soft_a, soft_b)} where values are
            continuous digit vectors [B, S, K]
        gt_digits_a: [B, K] ground-truth digit vector for operand A (float)
        gt_digits_b: [B, K] ground-truth digit vector for operand B (float)
        has_aux: [B] boolean mask (True for examples with valid aux targets)
        attention_mask: [B, S] (1 = real token, 0 = padding)
        eq_positions: [B] token index of the '=' sign
        op_positions: [B] (unused, kept for API compatibility)

    Returns:
        Scalar MSE loss averaged over valid entries, or zero if none.
    """
    if not arb_extractions or not has_aux.any():
        for _layer_id, (soft_a, _soft_b) in arb_extractions.items():
            return (soft_a * 0).sum()
        return gt_digits_a.new_zeros((), requires_grad=True)

    B = has_aux.shape[0]

    # Gather soft digits at the '=' position for each example
    # eq_positions: [B] -> index into sequence dim
    total_loss = gt_digits_a.new_zeros(())
    num_layers = 0
    valid_mask = has_aux.float()  # [B]
    count = valid_mask.sum()

    if count == 0:
        for _layer_id, (soft_a, _soft_b) in arb_extractions.items():
            return (soft_a * 0).sum()

    for _layer_id, (soft_a, soft_b) in arb_extractions.items():
        # soft_a, soft_b: [B, S, K]
        K = soft_a.size(2)

        # Gather at eq_position: [B, K]
        idx = eq_positions.unsqueeze(1).unsqueeze(2).expand(B, 1, K)  # [B, 1, K]
        pred_a = soft_a.gather(1, idx).squeeze(1)  # [B, K]
        pred_b = soft_b.gather(1, idx).squeeze(1)  # [B, K]

        # MSE per example: [B, K] -> [B]
        mse_a = ((pred_a - gt_digits_a) ** 2).mean(dim=1)  # [B]
        mse_b = ((pred_b - gt_digits_b) ** 2).mean(dim=1)  # [B]

        # Masked average
        layer_loss = ((mse_a + mse_b) * valid_mask).sum() / count

        total_loss = total_loss + layer_loss
        num_layers += 1

    if num_layers > 0:
        total_loss = total_loss / num_layers

    return total_loss
