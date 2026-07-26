"""Tests for the fully-trainable toy-pretraining setup."""

import torch
import pytest

from mathllm.pretraining.arithmetic_bpe_tokenizer import ArithmeticBPETokenizer
from mathllm.pretraining.data import CONTEXTUAL_TRAINING_TEMPLATES, MixtureSpec, build_mixture, contextual_arithmetic_texts
from mathllm.pretraining.experiment import ToyExperimentConfig, build_model, resolve_device


def _tokenizer():
    return ArithmeticBPETokenizer.train(["Natural-language prose is compressed by BPE.\n", "12+3=15\n"], 128)


def test_bpe_tokenizer_keeps_arithmetic_symbols_standalone():
    tokenizer = _tokenizer()
    ids = tokenizer.encode("12+3=")

    assert len(ids) == 5
    assert tokenizer.decode(ids) == "12+3="
    assert ids == [tokenizer.token_to_id[character] for character in "12+3="]


def test_atomic_fact_tokens_are_one_token_and_never_digit_tokens():
    entity = "<factentityabcde>"
    value = "<factvaluevwxyz>"
    tokenizer = ArithmeticBPETokenizer.train(
        [f"{entity}<factmap>{value}\n", "35+23=58\n"],
        128,
        atomic_tokens=(entity, value, "<factmap>"),
    )
    assert len(tokenizer.encode(entity)) == 1
    assert len(tokenizer.encode(value)) == 1
    assert tokenizer.decode(tokenizer.encode(entity)) == entity
    assert len(tokenizer.encode("35+23=")) == 6

    config = ToyExperimentConfig()
    arb = build_model(config, "arb", tokenizer)
    entity_id = tokenizer.encode(entity)[0]
    assert arb.compute_core.extract.token_digit_value[entity_id].item() == -1


def test_mixture_has_exact_train_and_eval_block_ratios():
    tokenizer = _tokenizer()
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


def test_three_way_mixture_has_exact_direct_and_contextual_block_counts():
    tokenizer = _tokenizer()
    spec = MixtureSpec(
        context_length=8,
        train_blocks=20,
        eval_blocks=20,
        arithmetic_token_fraction=0.25,
        max_digits=1,
        invocation_fraction=0.0,
        direct_equation_token_fraction=0.15,
        contextual_equation_token_fraction=0.10,
        seed=7,
    )
    mixture = build_mixture(spec, ["A short prose document. " * 50], tokenizer)

    assert torch.bincount(mixture["train_sources"], minlength=3).tolist() == [15, 3, 2]
    assert torch.bincount(mixture["eval_sources"], minlength=3).tolist() == [15, 3, 2]
    assert mixture["metadata"]["mixture_type"] == "three_way_exact_blocks"


def test_contextual_source_uses_multiple_instruction_templates():
    texts = contextual_arithmetic_texts(count=128, seed=7, max_digits=1)

    used_templates = [template for template in CONTEXTUAL_TRAINING_TEMPLATES if any(text.startswith(template) for text in texts)]
    assert len(used_templates) > 1


def test_substantive_mixture_refuses_to_cycle_a_tiny_prose_source():
    tokenizer = _tokenizer()
    spec = MixtureSpec(
        context_length=8,
        train_blocks=40,
        eval_blocks=20,
        arithmetic_token_fraction=0.25,
        max_digits=1,
        invocation_fraction=0.0,
        direct_equation_token_fraction=0.15,
        contextual_equation_token_fraction=0.10,
        require_unique_source_blocks=True,
        seed=7,
    )

    with pytest.raises(ValueError, match="train prose has only"):
        build_mixture(spec, ["tiny prose"], tokenizer)


def test_toy_baseline_and_arb_are_fully_trainable():
    config = ToyExperimentConfig()
    tokenizer = _tokenizer()
    baseline = build_model(config, "baseline", tokenizer)
    arb = build_model(config, "arb", tokenizer)

    assert all(parameter.requires_grad for parameter in baseline.parameters())
    assert all(parameter.requires_grad for parameter in arb.base_model.parameters())
    assert sum(parameter.numel() for parameter in arb.parameters()) > sum(
        parameter.numel() for parameter in baseline.parameters()
    )


def test_capacity_matched_baseline_uses_configured_mlp_width():
    config = ToyExperimentConfig()
    config.model.baseline_n_inner = config.model.n_inner + 3
    tokenizer = _tokenizer()
    baseline = build_model(config, "baseline", tokenizer)
    arb = build_model(config, "arb", tokenizer)

    assert sum(parameter.numel() for parameter in baseline.parameters()) > sum(
        parameter.numel() for parameter in arb.base_model.parameters()
    )


def test_auto_device_is_a_supported_backend():
    assert str(resolve_device("auto")) in {"cpu", "cuda", "mps"}
