from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mathllm.config import Config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "benchmark_zero_forgetting.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("test_zero_forgetting_script_module", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero_forgetting_script_uses_configured_model_dir_and_base_model(monkeypatch, tmp_path):
    module = _load_script_module()

    model_dir = tmp_path / "trained_model_auto"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("evaluation: {}\n")

    yaml_config = Config()
    yaml_config.training.device = "cpu"
    yaml_config.training.base_model = "yaml-base-model"
    yaml_config.training.final_model_dir = str(model_dir)

    captured = {}

    class FakeTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"
            self.pad_token_id = 7

    class FakeHFModel:
        def __init__(self, name):
            self.name = name
            self.config = type("Cfg", (), {"pad_token_id": None, "eos_token_id": 7})()
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)
            return self

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            captured.setdefault("tokenizers", []).append(model_name)
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name):
            captured.setdefault("base_models", []).append(model_name)
            return FakeHFModel(model_name)

    def fake_from_exported_model(cls, output_dir, device=None):
        captured["arb_model_dir"] = output_dir
        return FakeHFModel("arb"), FakeTokenizer(), Config()

    def fake_benchmark(**kwargs):
        captured["benchmark_kwargs"] = kwargs
        return {"benchmarks": {}, "markdown_table": "| x |"}

    monkeypatch.setattr(module, "load_config", lambda path: yaml_config)
    monkeypatch.setattr(module, "get_device", lambda device: "cpu")
    monkeypatch.setattr(module, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(module, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(module.GPT2WithARB, "from_exported_model", classmethod(fake_from_exported_model))
    monkeypatch.setattr(module, "run_zero_forgetting_benchmark", fake_benchmark)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--config",
            "configs/360m.yaml",
            "--batch-size",
            "4",
            "--piqa-limit",
            "10",
            "--hellaswag-limit",
            "12",
            "--wikitext-limit",
            "8",
        ],
    )

    module.main()

    assert captured["base_models"] == ["yaml-base-model"]
    assert captured["tokenizers"] == ["yaml-base-model"]
    assert captured["arb_model_dir"] == str(model_dir)
    assert captured["benchmark_kwargs"]["batch_size"] == 4
    assert captured["benchmark_kwargs"]["piqa_limit"] == 10
    assert captured["benchmark_kwargs"]["hellaswag_limit"] == 12
    assert captured["benchmark_kwargs"]["wikitext_limit"] == 8
