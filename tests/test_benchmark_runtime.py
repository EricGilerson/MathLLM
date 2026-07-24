from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "benchmark_runtime.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("test_benchmark_runtime_module", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_in_process_cpu_calculator_handles_supported_operations():
    module = _load_script_module()

    assert module.in_process_cpu_calculator("35 / 7 =") == "5"
    assert module.in_process_cpu_calculator("12 + 30 =") == "42"
    assert module.in_process_cpu_calculator("12 - 30 =") == "-18"
    assert module.in_process_cpu_calculator("12 * 30 =") == "360"
    with pytest.raises(ValueError, match="exact nonzero division"):
        module.in_process_cpu_calculator("7 / 2 =")


def test_arithmetic_prompt_suite_is_deterministic_and_calculator_valid():
    module = _load_script_module()

    prompts = module.arithmetic_prompts(12, seed=9)

    assert prompts == module.arithmetic_prompts(12, seed=9)
    assert {prompt.split()[1] for prompt, _ in prompts} == {"+", "-", "*", "/"}
    assert all(module.in_process_cpu_calculator(prompt) == expected for prompt, expected in prompts)


def test_summary_reports_tail_latency_percentiles():
    module = _load_script_module()

    summary = module._summary([{"latency_seconds": value} for value in (1.0, 2.0, 3.0, 4.0)])

    assert summary["latency_p90_seconds"] == 4.0
    assert summary["latency_p95_seconds"] == 4.0
