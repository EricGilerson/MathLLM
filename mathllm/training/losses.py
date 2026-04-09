"""Auxiliary losses for ARB training.

The key auxiliary loss directly supervises digit extraction (Stage 1) against
ground-truth operand digits using per-digit cross-entropy classification.

Head A is supervised at the operator position (where operand A is most recent
in the causal window), and head B at the '=' position (where operand B is
most recent). This eliminates the recency bias that otherwise makes A harder
to extract than B.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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
    """Compute cross-entropy loss between extracted logits and ground-truth digits.

    Head A is supervised at op_position (operator token, where A is most recent).
    Head B is supervised at eq_position ('=' token, where B is most recent).
    If op_positions is not provided, both heads use eq_position.

    Args:
        arb_extractions: {layer_id: (logits_a, logits_b)} where values are
            classification logits [B, S, K, C]
        gt_digits_a: [B, K] ground-truth digit vector for operand A (float)
        gt_digits_b: [B, K] ground-truth digit vector for operand B (float)
        has_aux: [B] boolean mask (True for examples with valid aux targets)
        attention_mask: [B, S] (1 = real token, 0 = padding)
        eq_positions: [B] token index of the '=' sign
        op_positions: [B] token index of the operator (+, -, *, etc.)

    Returns:
        Scalar cross-entropy loss averaged over valid entries, or zero if none.
    """
    if not arb_extractions or not has_aux.any():
        for _layer_id, (logits_a, _logits_b) in arb_extractions.items():
            return (logits_a * 0).sum()
        return gt_digits_a.new_zeros((), requires_grad=True)

    B, S = attention_mask.shape

    # Target digits as long for cross-entropy: [B, K]
    target_a = gt_digits_a.long()
    target_b = gt_digits_b.long()

    # Position masks: head A at operator, head B at '='
    positions = torch.arange(S, device=attention_mask.device).unsqueeze(0)  # [1, S]
    pos_a = op_positions if op_positions is not None else eq_positions
    mask_a = (
        has_aux.float().unsqueeze(1)
        * attention_mask.float()
        * (positions == pos_a.unsqueeze(1)).float()
    )  # [B, S]
    mask_b = (
        has_aux.float().unsqueeze(1)
        * attention_mask.float()
        * (positions == eq_positions.unsqueeze(1)).float()
    )  # [B, S]

    total_loss = gt_digits_a.new_zeros(())
    num_layers = 0

    for _layer_id, (logits_a, logits_b) in arb_extractions.items():
        # logits_a, logits_b: [B, S, K, C]
        K = logits_a.size(2)
        C = logits_a.size(3)

        # Broadcast targets across sequence positions: [B, K] -> [B, S, K]
        tgt_a = target_a.unsqueeze(1).expand(B, S, K)
        tgt_b = target_b.unsqueeze(1).expand(B, S, K)

        # Flatten for cross_entropy: [B*S*K, C] and [B*S*K]
        ce_a = F.cross_entropy(
            logits_a.reshape(-1, C), tgt_a.reshape(-1), reduction="none"
        ).view(B, S, K)
        ce_b = F.cross_entropy(
            logits_b.reshape(-1, C), tgt_b.reshape(-1), reduction="none"
        ).view(B, S, K)

        # Apply separate masks per head: [B, S, 1] broadcast over K digits
        masked_a = ce_a * mask_a.unsqueeze(2)
        masked_b = ce_b * mask_b.unsqueeze(2)

        count_a = mask_a.sum() * K
        count_b = mask_b.sum() * K
        layer_loss = gt_digits_a.new_zeros(())
        if count_a > 0:
            layer_loss = layer_loss + masked_a.sum() / count_a
        if count_b > 0:
            layer_loss = layer_loss + masked_b.sum() / count_b

        total_loss = total_loss + layer_loss
        num_layers += 1

    if num_layers > 0:
        total_loss = total_loss / num_layers

    return total_loss
