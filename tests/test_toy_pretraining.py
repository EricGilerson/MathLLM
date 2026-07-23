"""Tests for the fully-trainable toy-pretraining setup."""

import torch

from mathllm.pretraining.char_tokenizer import CharTokenizer
from mathllm.pretraining.data import MixtureSpec, build_mixture
from mathllm.pretraining.experiment import ToyExperimentConfig, build_model


def test_character_tokenizer_keeps_arithmetic_symbols_standalone():
    tokenizer = CharTokenizer()
    ids = tokenizer.encode("12+3=")

    assert len(ids) == 5
    assert tokenizer.decode(ids) == "12+3="


def test_mixture_has_exact_train_and_eval_block_ratios():
    tokenizer = CharTokenizer()
    spec = MixtureSpec(
        context_length=8,
        train_blocks=12,
        eval_blocks=4,
        arithmetic_token_fraction=0.25,
        max_digits=1,
        invocation_fraction=0.25,
        seed=7,
    )
    mixture = build_mixture(spec, ["A short prose document. " * 20], tokenizer)

    assert int(mixture["train_sources"].sum()) == 3
    assert int(mixture["eval_sources"].sum()) == 1
    assert mixture["train_input_ids"].shape == (12, 9)
    assert mixture["eval_input_ids"].shape == (4, 9)


def test_toy_baseline_and_arb_are_fully_trainable():
    config = ToyExperimentConfig()
    tokenizer = CharTokenizer()
    baseline = build_model(config, "baseline", tokenizer)
    arb = build_model(config, "arb", tokenizer)

    assert all(parameter.requires_grad for parameter in baseline.parameters())
    assert all(parameter.requires_grad for parameter in arb.base_model.parameters())
    assert sum(parameter.numel() for parameter in arb.parameters()) > sum(
        parameter.numel() for parameter in baseline.parameters()
    )
