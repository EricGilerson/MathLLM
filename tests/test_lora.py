"""Tests for evaluation-time LoRA scaling."""

import torch
import torch.nn as nn

from mathllm.model.lora import LoRALinear


def test_lora_eval_multiplier_scales_adapter_contribution():
    base_linear = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        base_linear.weight.copy_(torch.eye(2))

    lora = LoRALinear(base_linear, rank=1, alpha=1.0)
    with torch.no_grad():
        lora.lora_B.copy_(torch.tensor([[1.0, 0.0]]))
        lora.lora_A.copy_(torch.tensor([[2.0], [4.0]]))

    x = torch.tensor([[3.0, 5.0]])

    out_full = lora(x)
    lora.set_eval_multiplier(0.5)
    out_half = lora(x)
    lora.set_eval_multiplier(0.0)
    out_zero = lora(x)

    base_out = base_linear(x)

    assert torch.allclose(out_half - base_out, (out_full - base_out) * 0.5, atol=1e-6)
    assert torch.allclose(out_zero, base_out, atol=1e-6)
