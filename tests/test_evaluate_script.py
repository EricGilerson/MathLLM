from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from mathllm.config import Config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate.py"


def _load_evaluate_module():
    spec = importlib.util.spec_from_file_location("test_evaluate_script_module", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evaluate_script_uses_yaml_eval_config_with_model_dir(monkeypatch):
    module = _load_evaluate_module()

    yaml_config = Config()
    yaml_config.training.device = "cpu"
    yaml_config.evaluation.num_samples_per_config = 1000
    yaml_config.evaluation.max_digits_range = (1, 3)
    yaml_config.evaluation.max_new_tokens = 15

    exported_config = Config()
    exported_config.evaluation.num_samples_per_config = 200
    exported_config.evaluation.max_digits_range = (1, 3)
    exported_config.evaluation.max_new_tokens = 15

    captured = {}

    class FakeEvaluator:
        def __init__(self, model, tokenizer, config, device=None):
            captured["evaluation_config"] = config

        def full_evaluation(
            self,
            eval_texts=None,
            include_prior_logit_analysis: bool = False,
            base_model=None,
            base_tokenizer=None,
        ):
            return {"ok": True}

    monkeypatch.setattr(module, "load_config", lambda path: yaml_config)
    monkeypatch.setattr(module, "get_device", lambda device: "cpu")
    monkeypatch.setattr(
        module.GPT2WithARB,
        "from_exported_model",
        classmethod(lambda cls, output_dir, device=None: ("model", "tokenizer", exported_config)),
    )
    monkeypatch.setattr(module, "ARBEvaluator", FakeEvaluator)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--config",
            "configs/360m.yaml",
            "--model-dir",
            "trained_model_360m",
        ],
    )

    module.main()

    eval_config = captured["evaluation_config"]
    assert eval_config.num_samples_per_config == 1000
    assert eval_config.max_digits_range == (1, 3)
    assert eval_config.max_new_tokens == 15


def test_evaluate_script_uses_configured_final_model_dir_by_default(monkeypatch, tmp_path):
    module = _load_evaluate_module()

    model_dir = tmp_path / "trained_model_auto"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("evaluation: {}\n")

    yaml_config = Config()
    yaml_config.training.device = "cpu"
    yaml_config.training.final_model_dir = str(model_dir)

    captured = {}

    class FakeEvaluator:
        def __init__(self, model, tokenizer, config, device=None):
            captured["evaluation_config"] = config

        def full_evaluation(
            self,
            eval_texts=None,
            include_prior_logit_analysis: bool = False,
            base_model=None,
            base_tokenizer=None,
        ):
            return {"ok": True}

    def fake_from_exported_model(cls, output_dir, device=None):
        captured["model_dir"] = output_dir
        return "model", "tokenizer", Config()

    monkeypatch.setattr(module, "load_config", lambda path: yaml_config)
    monkeypatch.setattr(module, "get_device", lambda device: "cpu")
    monkeypatch.setattr(module.GPT2WithARB, "from_exported_model", classmethod(fake_from_exported_model))
    monkeypatch.setattr(module, "ARBEvaluator", FakeEvaluator)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--config",
            "configs/360m.yaml",
        ],
    )

    module.main()

    assert captured["model_dir"] == str(model_dir)
    assert captured["evaluation_config"] is yaml_config.evaluation


def test_evaluate_script_base_model_only_bypasses_exported_model(monkeypatch, tmp_path):
    module = _load_evaluate_module()

    model_dir = tmp_path / "trained_model_auto"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("evaluation: {}\n")

    yaml_config = Config()
    yaml_config.training.device = "cpu"
    yaml_config.training.base_model = "test-base-model"
    yaml_config.training.final_model_dir = str(model_dir)

    captured = {}

    class FakeTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"
            self.pad_token_id = 7

    class FakeHFModel:
        def __init__(self):
            self.config = type("Cfg", (), {"pad_token_id": None, "eos_token_id": 7})()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name):
            captured["base_model_name"] = model_name
            return FakeHFModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            captured["tokenizer_name"] = model_name
            return FakeTokenizer()

    class FakeEvaluator:
        def __init__(self, model, tokenizer, config, device=None):
            captured["model"] = model
            captured["tokenizer"] = tokenizer
            captured["evaluation_config"] = config

        def full_evaluation(
            self,
            eval_texts=None,
            include_prior_logit_analysis: bool = False,
            base_model=None,
            base_tokenizer=None,
        ):
            return {"ok": True}

    monkeypatch.setattr(module, "load_config", lambda path: yaml_config)
    monkeypatch.setattr(module, "get_device", lambda device: "cpu")
    monkeypatch.setattr(module, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(module, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(
        module.GPT2WithARB,
        "from_exported_model",
        classmethod(lambda cls, output_dir, device=None: (_ for _ in ()).throw(AssertionError("should not load exported model"))),
    )
    monkeypatch.setattr(module, "ARBEvaluator", FakeEvaluator)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--config",
            "configs/360m.yaml",
            "--base-model-only",
        ],
    )

    module.main()

    assert captured["base_model_name"] == "test-base-model"
    assert captured["tokenizer_name"] == "test-base-model"
    assert isinstance(captured["model"], module.BaseModelEvaluatorAdapter)
    assert captured["model"].model.config.pad_token_id == 7
    assert captured["evaluation_config"] is yaml_config.evaluation


def test_evaluate_script_prior_logit_analysis_loads_frozen_base_model(monkeypatch):
    module = _load_evaluate_module()

    yaml_config = Config()
    yaml_config.training.device = "cpu"
    yaml_config.training.base_model = "yaml-base-model"

    exported_config = Config()
    exported_config.training.base_model = "yaml-base-model"

    captured = {"tokenizer_names": [], "model_names": []}

    class FakeTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"
            self.pad_token_id = 7

    class FakeHFModel:
        def __init__(self):
            self.config = type("Cfg", (), {"pad_token_id": None, "eos_token_id": 7})()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            captured["tokenizer_names"].append(model_name)
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name):
            captured["model_names"].append(model_name)
            return FakeHFModel()

    class FakeEvaluator:
        def __init__(self, model, tokenizer, config, device=None):
            captured["evaluation_config"] = config

        def full_evaluation(
            self,
            eval_texts=None,
            include_prior_logit_analysis: bool = False,
            base_model=None,
            base_tokenizer=None,
        ):
            captured["include_prior_logit_analysis"] = include_prior_logit_analysis
            captured["base_model"] = base_model
            captured["base_tokenizer"] = base_tokenizer
            return {"ok": True}

    monkeypatch.setattr(module, "load_config", lambda path: yaml_config)
    monkeypatch.setattr(module, "get_device", lambda device: "cpu")
    monkeypatch.setattr(module, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(module, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(
        module.GPT2WithARB,
        "from_exported_model",
        classmethod(lambda cls, output_dir, device=None: ("arb-model", "arb-tokenizer", exported_config)),
    )
    monkeypatch.setattr(module, "ARBEvaluator", FakeEvaluator)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--config",
            "configs/360m.yaml",
            "--model-dir",
            "trained_model_360m",
            "--prior-logit-analysis",
        ],
    )

    module.main()

    assert captured["include_prior_logit_analysis"] is True
    assert captured["tokenizer_names"] == ["yaml-base-model"]
    assert captured["model_names"] == ["yaml-base-model"]
    assert captured["base_model"].config.pad_token_id == 7


def test_evaluate_script_rejects_prior_logit_analysis_for_base_model_only(monkeypatch):
    module = _load_evaluate_module()

    yaml_config = Config()
    yaml_config.training.device = "cpu"

    monkeypatch.setattr(module, "load_config", lambda path: yaml_config)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--config",
            "configs/360m.yaml",
            "--base-model-only",
            "--prior-logit-analysis",
        ],
    )

    with pytest.raises(SystemExit):
        module.main()


def test_evaluate_script_rejects_mismatched_exported_base_model_for_prior_analysis(monkeypatch):
    module = _load_evaluate_module()

    yaml_config = Config()
    yaml_config.training.device = "cpu"
    yaml_config.training.base_model = "yaml-base-model"

    exported_config = Config()
    exported_config.training.base_model = "exported-base-model"

    class FakeEvaluator:
        def __init__(self, model, tokenizer, config, device=None):
            pass

        def full_evaluation(
            self,
            eval_texts=None,
            include_prior_logit_analysis: bool = False,
            base_model=None,
            base_tokenizer=None,
        ):
            return {"ok": True}

    monkeypatch.setattr(module, "load_config", lambda path: yaml_config)
    monkeypatch.setattr(module, "get_device", lambda device: "cpu")
    monkeypatch.setattr(
        module.GPT2WithARB,
        "from_exported_model",
        classmethod(lambda cls, output_dir, device=None: ("arb-model", "arb-tokenizer", exported_config)),
    )
    monkeypatch.setattr(module, "ARBEvaluator", FakeEvaluator)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--config",
            "configs/360m.yaml",
            "--model-dir",
            "trained_model_360m",
            "--prior-logit-analysis",
        ],
    )

    with pytest.raises(SystemExit):
        module.main()
