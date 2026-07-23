"""Regression tests for the no-trigger native-model fast path."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mathllm.config import Config
from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.pretraining.arithmetic_bpe_tokenizer import ArithmeticBPETokenizer


def _tiny_model():
    tokenizer = ArithmeticBPETokenizer.train(
        ["ordinary prose remains ordinary prose\n", "2+3=5\n"], vocab_size=128,
    )
    base = GPT2LMHeadModel(GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=128,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))
    config = Config()
    config.rns.num_digit_slots = 4
    config.arb.layer_positions = (0, 1)
    config.arb.lora_rank = 0
    model = GPT2WithARB(config, base_model=base)
    model.build_token_digit_tables(tokenizer)
    model.eval()
    return model, base, tokenizer


def test_no_trigger_uses_native_base_forward_and_preserves_kv_cache(monkeypatch):
    model, base, tokenizer = _tiny_model()
    input_ids = tokenizer.encode("ordinary prose", return_tensors="pt")
    original_detect = model.compute_core.extract.find_valid_equations
    detection_calls = 0

    def count_detection(*args, **kwargs):
        nonlocal detection_calls
        detection_calls += 1
        return original_detect(*args, **kwargs)

    def fail_compute(*args, **kwargs):
        raise AssertionError("No-trigger path should not invoke ARBComputeCore.forward")

    monkeypatch.setattr(model.compute_core.extract, "find_valid_equations", count_detection)
    monkeypatch.setattr(model.compute_core, "forward", fail_compute)
    with torch.inference_mode():
        wrapped = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
        direct = base(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True, return_dict=True)

    torch.testing.assert_close(wrapped["logits"], direct.logits)
    assert detection_calls == 1
    assert wrapped["arb_extractions"] == {}
    assert wrapped["past_key_values"] is not None

    next_ids = tokenizer.encode(".", return_tensors="pt")
    full_mask = torch.ones((1, input_ids.size(1) + next_ids.size(1)), dtype=torch.long)
    with torch.inference_mode():
        cached = model(
            input_ids=next_ids,
            attention_mask=full_mask,
            past_key_values=wrapped["past_key_values"],
            use_cache=True,
        )
    assert cached["past_key_values"] is not None
    assert cached["arb_extractions"] == {}
    assert detection_calls == 2


def test_valid_equation_keeps_custom_path_during_cached_generation():
    model, _, tokenizer = _tiny_model()
    prompt = tokenizer.encode("2+3=", return_tensors="pt")
    model.compute_core.enter_generation_mode()
    model._generation_mode = True
    try:
        with torch.inference_mode():
            first = model(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=True)
        assert first["arb_extractions"]
        assert bool(model.compute_core._cached_has_eq.any())

        digit = torch.tensor([[tokenizer.token_to_id["5"]]], dtype=torch.long)
        full_mask = torch.ones((1, prompt.size(1) + 1), dtype=torch.long)
        with torch.inference_mode():
            second = model(
                input_ids=digit,
                attention_mask=full_mask,
                past_key_values=first["past_key_values"],
                use_cache=True,
            )
        assert second["arb_extractions"]
    finally:
        model.compute_core.exit_generation_mode()
        model._generation_mode = False
