"""Tests for checkpoint-based inference debugging helpers."""

from __future__ import annotations

import torch
from torch import nn

from mathllm.config import Config
from scripts import infer_checkpoint


def _make_logits_for_digits(digits: list[int], *, num_classes: int = 10) -> torch.Tensor:
    """Build a [K, C] logit tensor whose argmax matches the requested digits."""
    logits = torch.full((len(digits), num_classes), -10.0)
    for idx, digit in enumerate(digits):
        logits[idx, digit] = 10.0
    return logits


class _FakeTokenizer:
    """Tokenizer stub for checkpoint loader tests."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, _name: str) -> _FakeTokenizer:
        return cls()


class _FakeARB(nn.Module):
    """Minimal ARB submodule with loadable state."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


class _FakeModel(nn.Module):
    """Minimal model surface for checkpoint loading."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.arbs = nn.ModuleDict({"4": _FakeARB()})
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class TestInferCheckpointHelpers:
    def test_collect_layer_extractions_decodes_last_token_logits(self):
        logits_a_layer4 = torch.stack(
            [
                _make_logits_for_digits([0, 0, 0]),
                _make_logits_for_digits([7, 4, 3]),
            ],
            dim=0,
        ).unsqueeze(0)
        logits_b_layer4 = torch.stack(
            [
                _make_logits_for_digits([0, 0, 0]),
                _make_logits_for_digits([1, 9, 2]),
            ],
            dim=0,
        ).unsqueeze(0)
        logits_a_layer8 = torch.stack(
            [
                _make_logits_for_digits([0, 0, 0]),
                _make_logits_for_digits([5, 5, 0]),
            ],
            dim=0,
        ).unsqueeze(0)
        logits_b_layer8 = torch.stack(
            [
                _make_logits_for_digits([0, 0, 0]),
                _make_logits_for_digits([2, 1, 0]),
            ],
            dim=0,
        ).unsqueeze(0)

        extractions = infer_checkpoint.collect_layer_extractions(
            {
                8: (logits_a_layer8, logits_b_layer8),
                4: (logits_a_layer4, logits_b_layer4),
            },
            token_index=1,
        )

        assert [item.layer_id for item in extractions] == [4, 8]
        assert extractions[0].digits_a == [7, 4, 3]
        assert extractions[0].digits_b == [1, 9, 2]
        assert extractions[0].value_a == 347
        assert extractions[0].value_b == 291
        assert extractions[1].value_a == 55
        assert extractions[1].value_b == 12

    def test_format_extractions_renders_compact_summary(self):
        formatted = infer_checkpoint.format_extractions(
            [
                infer_checkpoint.LayerExtraction(
                    layer_id=4,
                    token_index=5,
                    digits_a=[7, 4, 3, 0],
                    digits_b=[1, 9, 2, 0],
                    value_a=347,
                    value_b=291,
                )
            ]
        )

        assert "layer 4 @ token 5" in formatted
        assert "a_digits=[7, 4, 3, 0]" in formatted
        assert "b_value=291" in formatted

    def test_resolve_checkpoint_path_uses_latest_when_unspecified(self, tmp_path):
        config = Config()
        config.training.checkpoint_dir = str(tmp_path)

        latest = tmp_path / "arb_latest.pt"
        latest.write_bytes(b"checkpoint")

        resolved = infer_checkpoint.resolve_checkpoint_path(config, None)
        assert resolved == latest

    def test_load_checkpointed_model_loads_arb_state(self, monkeypatch, tmp_path):
        checkpoint_path = tmp_path / "arb_latest.pt"
        checkpoint_path.write_bytes(b"placeholder")

        config = Config()
        config.training.base_model = "fake-gpt2"
        config.training.checkpoint_dir = str(tmp_path)

        monkeypatch.setattr(infer_checkpoint, "load_config", lambda _path: config)
        monkeypatch.setattr(infer_checkpoint, "GPT2Tokenizer", _FakeTokenizer)
        monkeypatch.setattr(infer_checkpoint, "GPT2WithARB", _FakeModel)
        monkeypatch.setattr(
            infer_checkpoint.torch,
            "load",
            lambda _path, map_location=None, weights_only=False: {
                "arb_state": {"4": {"weight": torch.tensor([2.5])}}
            },
        )

        model, tokenizer, loaded_config, resolved = infer_checkpoint.load_checkpointed_model(
            config_path="configs/default.yaml",
            checkpoint_path=str(checkpoint_path),
            device=torch.device("cpu"),
        )

        assert isinstance(tokenizer, _FakeTokenizer)
        assert tokenizer.pad_token == tokenizer.eos_token
        assert loaded_config is config
        assert resolved == checkpoint_path
        assert model.moved_to == torch.device("cpu")
        torch.testing.assert_close(model.arbs["4"].weight.detach(), torch.tensor([2.5]))
