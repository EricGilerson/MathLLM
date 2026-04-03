"""Integration tests: full model forward/backward pass.

These tests require downloading GPT-2, so they are marked with pytest.mark.slow.
Run with: pytest tests/test_integration.py -m slow
"""

import pytest
import torch

from mathllm.config import Config, load_config


@pytest.fixture
def config():
    """Minimal config for testing."""
    cfg = Config()
    cfg.training.base_model = "gpt2"
    cfg.arb.layer_positions = (4, 8, 10)
    cfg.rns.num_digit_slots = 10
    return cfg


@pytest.mark.slow
class TestModelIntegration:
    def test_forward_pass(self, config):
        """Model should produce logits and loss."""
        from mathllm.model.gpt2_arb import GPT2WithARB

        model = GPT2WithARB(config)
        input_ids = torch.tensor([[50, 347, 12, 291, 796]])
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (1, 5, 50257)  # GPT-2 vocab size
        assert outputs["loss"].item() > 0

    def test_zero_init_matches_base(self, config):
        """With zero W_proj, output should match unmodified GPT-2."""
        from transformers import GPT2LMHeadModel

        from mathllm.model.gpt2_arb import GPT2WithARB

        model = GPT2WithARB(config)
        base = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
        base.eval()
        model.eval()

        input_ids = torch.tensor([[50, 347, 12, 291, 796]])

        with torch.no_grad():
            arb_logits = model(input_ids=input_ids)["logits"]
            base_logits = base(input_ids=input_ids).logits

        assert torch.allclose(arb_logits, base_logits, atol=1e-4), \
            f"Max diff: {(arb_logits - base_logits).abs().max().item()}"

    def test_gradient_flow(self, config):
        """ARB parameters should receive gradients after backward pass."""
        from mathllm.model.gpt2_arb import GPT2WithARB

        model = GPT2WithARB(config)
        # Set W_proj non-zero to enable gradient flow
        for arb in model.arbs.values():
            with torch.no_grad():
                arb.inject.projection.weight.fill_(0.001)

        input_ids = torch.tensor([[50, 347, 12, 291, 796]])
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        outputs["loss"].backward()

        # Check that extraction weights have gradients
        for key, arb in model.arbs.items():
            assert arb.extract.W_a.weight.grad is not None, \
                f"ARB {key}: W_a has no gradient"
            assert arb.extract.W_a.weight.grad.abs().sum() > 0, \
                f"ARB {key}: W_a gradient is zero"

    def test_only_arb_params_trainable(self, config):
        """Only ARB learned params should have requires_grad=True."""
        from mathllm.model.gpt2_arb import GPT2WithARB

        model = GPT2WithARB(config)
        trainable = model.get_trainable_parameters()
        assert len(trainable) > 0

        # Base model should be frozen
        for param in model.base_model.parameters():
            assert not param.requires_grad

    def test_generate(self, config):
        """Model should be able to generate tokens."""
        from mathllm.model.gpt2_arb import GPT2WithARB

        model = GPT2WithARB(config)
        model.eval()
        input_ids = torch.tensor([[50, 347]])

        output = model.generate(input_ids, max_new_tokens=5, greedy=True)
        assert output.shape[1] == 2 + 5  # prompt + generated

    def test_attention_mask(self, config):
        """Model should handle attention masks correctly."""
        from mathllm.model.gpt2_arb import GPT2WithARB

        model = GPT2WithARB(config)
        input_ids = torch.tensor([[50, 347, 12, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        assert outputs["loss"].item() > 0
