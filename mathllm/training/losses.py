"""Auxiliary losses for ARB training.

The key auxiliary loss directly supervises digit extraction (Stage 1) against
ground-truth operand digits. This breaks the chicken-and-egg problem where
extraction and injection must learn simultaneously from weak LM loss.

The loss is computed on continuous (pre-round) extraction outputs, giving
smooth proportional gradients rather than quantized integer-valued ones.
A position-weight ramp emphasises later sequence positions, which have
seen both operands via causal attention.
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
) -> Tensor:
    """Compute MSE loss between extracted and ground-truth digit vectors.

    The ground-truth digits are broadcast across all sequence positions —
    at every token, the ARB should learn to extract the example's operands.
    A position-weight ramp (0→1 across the sequence) down-weights early
    positions that cannot yet see both operands due to causal attention.

    Args:
        arb_extractions: {layer_id: (d_a_cont, d_b_cont)} where values are
            continuous (pre-round) extractions [B, S, K]
        gt_digits_a: [B, K] ground-truth digit vector for operand A
        gt_digits_b: [B, K] ground-truth digit vector for operand B
        has_aux: [B] boolean mask (True for examples with valid aux targets)
        attention_mask: [B, S] (1 = real token, 0 = padding)

    Returns:
        Scalar MSE loss averaged over valid entries, or zero if no valid entries.
    """
    if not arb_extractions or not has_aux.any():
        # Return a zero that participates in the graph so backward() works
        # when this is the only loss component (Phase 1 with no aux examples).
        for _layer_id, (d_a, _d_b) in arb_extractions.items():
            return (d_a * 0).sum()
        return gt_digits_a.new_zeros((), requires_grad=True)

    # [B, 1, K] — broadcast target across sequence positions
    target_a = gt_digits_a.unsqueeze(1)
    target_b = gt_digits_b.unsqueeze(1)

    # [B, S, 1] — mask: valid example AND non-padding token
    mask = (has_aux.unsqueeze(1) & attention_mask.bool()).unsqueeze(2).float()

    # Position weight: ramp from 0→1 across sequence length.
    # Later positions have seen more context via causal attention and are
    # more likely to have both operands available for extraction.
    seq_len = attention_mask.size(1)
    pos_weight = torch.linspace(0.0, 1.0, seq_len, device=mask.device)
    pos_weight = pos_weight[None, :, None]  # [1, S, 1]
    mask = mask * pos_weight

    total_loss = gt_digits_a.new_zeros(())
    num_layers = 0

    for _layer_id, (d_a, d_b) in arb_extractions.items():
        # d_a, d_b: [B, S, K] — continuous (pre-round) values
        err_a = (d_a - target_a) ** 2  # [B, S, K]
        err_b = (d_b - target_b) ** 2  # [B, S, K]

        # Mask and average
        masked_err = (err_a + err_b) * mask  # [B, S, K]
        count = mask.sum() * d_a.size(-1)  # total valid (position, digit) entries
        if count > 0:
            total_loss = total_loss + masked_err.sum() / count
        num_layers += 1

    if num_layers > 0:
        total_loss = total_loss / num_layers

    return total_loss

