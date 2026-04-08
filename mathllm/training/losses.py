"""Auxiliary losses for ARB training.

The key auxiliary loss directly supervises digit extraction (Stage 1) against
ground-truth operand digits using per-digit cross-entropy classification.

The loss is computed on classification logits, giving strong gradient signal
for discrete digit prediction. A binary position mask restricts supervision
to sequence positions at or after eq_position, where both operands have been
seen via causal attention.
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
) -> Tensor:
    """Compute cross-entropy loss between extracted logits and ground-truth digits.

    Only positions at or after eq_position are supervised — earlier positions
    cannot have seen both operands due to causal attention.

    Args:
        arb_extractions: {layer_id: (logits_a, logits_b)} where values are
            classification logits [B, S, K, C]
        gt_digits_a: [B, K] ground-truth digit vector for operand A (float)
        gt_digits_b: [B, K] ground-truth digit vector for operand B (float)
        has_aux: [B] boolean mask (True for examples with valid aux targets)
        attention_mask: [B, S] (1 = real token, 0 = padding)
        eq_positions: [B] token index where both operands are visible

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

    # Build position mask: supervise only at eq_position (the '=' token).
    # This is the cleanest extraction point — both operands are visible via
    # causal attention but answer tokens have not yet appeared.
    positions = torch.arange(S, device=attention_mask.device).unsqueeze(0)  # [1, S]
    pos_mask = (positions == eq_positions.unsqueeze(1)).float()  # [B, S]

    # Combined mask: has_aux AND non-padding AND position >= eq_position
    mask = (
        has_aux.float().unsqueeze(1)        # [B, 1]
        * attention_mask.float()             # [B, S]
        * pos_mask                           # [B, S]
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

        # Apply mask: [B, S, 1] broadcast over K digits
        masked_ce = (ce_a + ce_b) * mask.unsqueeze(2)  # [B, S, K]
        count = mask.sum() * K
        if count > 0:
            total_loss = total_loss + masked_ce.sum() / count
        num_layers += 1

    if num_layers > 0:
        total_loss = total_loss / num_layers

    return total_loss
