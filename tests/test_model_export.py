"""Tests for standalone model export bundles."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mathllm.config import Config
from mathllm.model import gpt2_arb


class FakeTokenizer:
    """Tokenizer stub that persists a minimal marker file."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"

    def encode(self, text, add_special_tokens=True):
        """Stub encode that returns token 28 for '='."""
        if text == "=":
            return [28]
        return [0]

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tokenizer.json").write_text("{}")

    @classmethod
    def from_pretrained(cls, output_dir: str | Path) -> FakeTokenizer:
        assert (Path(output_dir) / "tokenizer.json").exists()
        return cls()


class FakeBaseModelConfig:
    """Minimal GPT-2 config stub for export tests."""

    model_type = "gpt2"

    def __init__(self, n_embd: int = 4):
        self.n_embd = n_embd
        self._attn_implementation = None

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text("{}")

    @classmethod
    def from_pretrained(cls, output_dir: str | Path) -> FakeBaseModelConfig:
        assert (Path(output_dir) / "config.json").exists()
        return cls()


class FakeBaseModel(nn.Module):
    """Base model stub with a single learnable weight."""

    def __init__(self, config: FakeBaseModelConfig | None = None):
        super().__init__()
        self.config = config or FakeBaseModelConfig()
        self.base_weight = nn.Parameter(torch.tensor([1.5]))

    @classmethod
    def from_config(cls, config):
        return cls(config)


class ExportOnlyModel(gpt2_arb.TransformerWithARB):
    """Minimal subclass that reuses the export helpers."""

    def __init__(self, config: Config, base_model: FakeBaseModel | None = None, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.base_model = base_model or FakeBaseModel()
        self.extra = nn.Linear(1, 1)


class TestModelExport:
    def test_export_bundle_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gpt2_arb, "AutoTokenizer", FakeTokenizer)
        monkeypatch.setattr(gpt2_arb, "AutoConfig", FakeBaseModelConfig)
        monkeypatch.setattr(gpt2_arb, "AutoModelForCausalLM", FakeBaseModel)

        config = Config()
        config.training.base_model = "gpt2"
        model = ExportOnlyModel(config)
        tokenizer = FakeTokenizer()

        with torch.no_grad():
            model.extra.weight.fill_(2.0)
            model.extra.bias.fill_(3.0)

        export_dir = model.save_exported_model(tmp_path / "bundle", tokenizer)
        assert (export_dir / gpt2_arb.EXPORT_STATE_FILENAME).exists()
        assert (export_dir / gpt2_arb.EXPORT_CONFIG_FILENAME).exists()
        assert (export_dir / gpt2_arb.EXPORT_BASE_MODEL_CONFIG_DIRNAME / "config.json").exists()
        assert (export_dir / "tokenizer.json").exists()

        reloaded_model, reloaded_tokenizer, reloaded_config = ExportOnlyModel.from_exported_model(
            export_dir
        )

        assert isinstance(reloaded_tokenizer, FakeTokenizer)
        assert reloaded_tokenizer.pad_token == reloaded_tokenizer.eos_token
        assert reloaded_config.training.final_model_dir == "trained_model/"
        torch.testing.assert_close(
            model.state_dict()["base_model.base_weight"],
            reloaded_model.state_dict()["base_model.base_weight"],
        )
        torch.testing.assert_close(
            model.state_dict()["extra.weight"],
            reloaded_model.state_dict()["extra.weight"],
        )
        torch.testing.assert_close(
            model.state_dict()["extra.bias"],
            reloaded_model.state_dict()["extra.bias"],
        )
