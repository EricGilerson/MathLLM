"""Tests for the target-masked capacity-displacement experiment."""

from pathlib import Path

import pytest
import torch

from mathllm.pretraining.capacity_displacement import (
    CONDITIONS,
    DisplacementDataConfig,
    DisplacementExperimentConfig,
    DisplacementTrainingConfig,
    _condition_model,
    prepare_displacement_data,
    recoverable_fact_bits,
    run_condition,
)
from mathllm.pretraining.arithmetic_bpe_tokenizer import ArithmeticBPETokenizer
from mathllm.pretraining.experiment import ToyModelConfig


def _tiny_config(tmp_path: Path) -> DisplacementExperimentConfig:
    return DisplacementExperimentConfig(
        model=ToyModelConfig(
            n_layer=2,
            n_embd=32,
            n_head=4,
            n_inner=64,
            baseline_n_inner=96,
        ),
        data=DisplacementDataConfig(
            dataset_file=str(tmp_path / "data.pt"),
            tokenizer_file=str(tmp_path / "tokenizer.json"),
            tokenizer_vocab_size=128,
            fact_count=16,
            fact_key_vocab_size=4,
            fact_value_vocab_size=8,
            max_digits=1,
            arithmetic_pool_size=8,
            arithmetic_eval_cases=4,
        ),
        training=DisplacementTrainingConfig(
            output_dir=str(tmp_path / "runs"),
            device="cpu",
            seed=31,
            model_seed=31,
            context_length=16,
            batch_size=4,
            max_steps=1,
            arithmetic_pretrain_steps=1,
            arithmetic_pretrain_eval_every=1,
            log_every=1,
            eval_every=1,
            curve_eval_cases=8,
            save_checkpoints=False,
        ),
    )


def test_dataset_masks_only_fact_values_and_arithmetic_answers(tmp_path):
    config = _tiny_config(tmp_path)
    data = prepare_displacement_data(config)

    assert data["fact_input_ids"].shape == (16, 16)
    assert data["arithmetic_input_ids"].shape == (8, 16)
    assert data["arithmetic_eval_input_ids"].shape == (4, 16)
    assert data["fact_labels"].ne(-100).sum(dim=1).unique().tolist() == [1]
    assert int(data["arithmetic_labels"].ne(-100).sum()) >= 8
    assert set(data["arithmetic_training_text"]).isdisjoint(data["arithmetic_eval_text"])
    assert data["metadata"]["mean_fact_exposures"] == pytest.approx(0.125)


def test_zero_arb_control_replaces_only_computed_result(tmp_path):
    config = _tiny_config(tmp_path)
    prepare_displacement_data(config)
    tokenizer = ArithmeticBPETokenizer.from_file(config.data.tokenizer_file)
    model = _condition_model(config, "zero_arb_mixed", tokenizer)
    ids = tokenizer.encode("7+2=9", return_tensors="pt")
    result = model.compute_core(ids, torch.ones_like(ids))

    assert result.has_eq.item()
    assert torch.count_nonzero(result.results).item() == 0


def test_recoverable_fact_bits_is_zero_at_chance_and_increases():
    assert recoverable_fact_bits(1 / 512, 25_000, 512) == 0.0
    assert recoverable_fact_bits(0.10, 25_000, 512) > 0.0
    assert recoverable_fact_bits(1.0, 25_000, 512) == pytest.approx(225_000.0)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_condition_smoke_trains_and_reports_metrics(tmp_path, condition):
    config = _tiny_config(tmp_path)
    data = prepare_displacement_data(config)
    metrics = run_condition(config, condition, data)

    assert metrics["condition"] == condition
    assert 0.0 <= metrics["fact_accuracy"] <= 1.0
    assert 0.0 <= metrics["arithmetic_exact_accuracy"] <= 1.0
    assert metrics["parameter_count"] > 0
    assert metrics["evaluation_history"][0]["step"] == 1
    if condition == "fact_only":
        assert metrics["arithmetic_pretrain_history"] == []
    else:
        assert metrics["arithmetic_pretrain_history"][0]["step"] == 1
    assert (tmp_path / "runs" / condition / "metrics.json").exists()
