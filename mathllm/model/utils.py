"""Model utilities: parameter counting, freezing, device management."""

from __future__ import annotations

import torch
import torch.nn as nn


def freeze_parameters(module: nn.Module) -> None:
    """Set requires_grad=False on all parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Count trainable and frozen parameters."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def get_device(preference: str = "auto") -> torch.device:
    """Resolve device preference to an actual torch.device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)
