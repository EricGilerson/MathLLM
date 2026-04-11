"""Tests for exporting standalone bundles from training checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from mathllm.config import Config
from scripts import export_checkpoint


class _FakeTokenizer:
    """Tokenizer placeholder for export helper tests."""


class _FakeModel:
    """Minimal exportable model stub."""

    def __init__(self):
        self.saved_calls: list[tuple[Path, object]] = []

    def save_exported_model(self, output_dir: str | Path, tokenizer) -> Path:
        output_dir = Path(output_dir)
        self.saved_calls.append((output_dir, tokenizer))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "bundle.marker").write_text("ok")
        return output_dir


class TestExportCheckpoint:
    def test_export_checkpoint_model_uses_config_final_model_dir(
        self, monkeypatch, tmp_path
    ):
        config = Config()
        config.training.final_model_dir = str(tmp_path / "trained_model")
        checkpoint_path = tmp_path / "arb_latest.pt"
        model = _FakeModel()
        tokenizer = _FakeTokenizer()

        monkeypatch.setattr(
            export_checkpoint,
            "load_checkpointed_model",
            lambda config_path, checkpoint_path, device: (
                model,
                tokenizer,
                config,
                Path(checkpoint_path),
            ),
        )

        export_dir, resolved_checkpoint = export_checkpoint.export_checkpoint_model(
            config_path="configs/default.yaml",
            checkpoint_path=str(checkpoint_path),
            output_dir=None,
            device=torch.device("cpu"),
        )

        assert export_dir == Path(config.training.final_model_dir)
        assert resolved_checkpoint == checkpoint_path
        assert model.saved_calls == [(Path(config.training.final_model_dir), tokenizer)]
        assert (export_dir / "bundle.marker").exists()

    def test_export_checkpoint_model_allows_output_override(
        self, monkeypatch, tmp_path
    ):
        config = Config()
        config.training.final_model_dir = str(tmp_path / "default_bundle")
        checkpoint_path = tmp_path / "arb_epoch_2.pt"
        output_dir = tmp_path / "custom_bundle"
        model = _FakeModel()
        tokenizer = _FakeTokenizer()

        monkeypatch.setattr(
            export_checkpoint,
            "load_checkpointed_model",
            lambda config_path, checkpoint_path, device: (
                model,
                tokenizer,
                config,
                Path(checkpoint_path),
            ),
        )

        export_dir, resolved_checkpoint = export_checkpoint.export_checkpoint_model(
            config_path="configs/default.yaml",
            checkpoint_path=str(checkpoint_path),
            output_dir=str(output_dir),
            device=torch.device("cpu"),
        )

        assert export_dir == output_dir
        assert resolved_checkpoint == checkpoint_path
        assert config.training.final_model_dir == str(output_dir)
        assert model.saved_calls == [(output_dir, tokenizer)]
        assert (export_dir / "bundle.marker").exists()
