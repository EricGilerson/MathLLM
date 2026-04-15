from __future__ import annotations

import math

import torch

from mathllm.evaluation.zero_forgetting import (
    MultipleChoiceExample,
    benchmark_multiple_choice,
    compute_perplexity,
    load_hellaswag_examples,
    load_piqa_examples,
    render_markdown_table,
    run_zero_forgetting_benchmark,
)


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]


class PreferenceModel:
    def __init__(self):
        self.config = type("Cfg", (), {"eos_token_id": 0})()
        self.eval_calls = 0
        self.to_calls: list[torch.device] = []
        self.prepare_calls: list[torch.device] = []

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.to_calls.append(torch.device(device))
        return self

    def prepare_for_device(self, device):
        self.prepare_calls.append(torch.device(device))

    def __call__(self, input_ids, attention_mask, labels=None):
        vocab_size = 256
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], vocab_size),
            0.0,
            dtype=torch.float32,
            device=input_ids.device,
        )

        for row in range(input_ids.shape[0]):
            seq_len = int(attention_mask[row].sum().item())
            for pos in range(seq_len - 1):
                next_token = int(input_ids[row, pos + 1].item())
                logits[row, pos, next_token] = next_token / 1000.0

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            valid_mask = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid_mask, 0)
            token_log_probs = torch.log_softmax(shift_logits, dim=-1).gather(
                dim=-1,
                index=safe_labels.unsqueeze(-1),
            ).squeeze(-1)
            nll = -(token_log_probs * valid_mask).sum()
            denom = valid_mask.sum().clamp_min(1)
            loss = nll / denom

        return {"logits": logits, "loss": loss}


def test_benchmark_multiple_choice_scores_sum_and_length_normalized_accuracy():
    model = PreferenceModel()
    tokenizer = FakeTokenizer()
    examples = [
        MultipleChoiceExample(prompt="Q1\nA:", choices=(" A", " B"), label=1),
        MultipleChoiceExample(prompt="Q2\nA:", choices=(" C", " A"), label=0),
    ]

    results = benchmark_multiple_choice(
        model,
        tokenizer,
        examples,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert results["num_examples"] == 2
    assert results["accuracy"] == 1.0
    assert results["length_normalized_accuracy"] == 1.0
    assert model.to_calls == [torch.device("cpu")]
    assert model.prepare_calls == [torch.device("cpu")]


def test_compute_perplexity_uses_token_stream_windows():
    model = PreferenceModel()
    tokenizer = FakeTokenizer()

    results = compute_perplexity(
        model,
        tokenizer,
        ["ABCD", "EFGH"],
        device=torch.device("cpu"),
        max_length=4,
        stride=2,
    )

    assert results["num_tokens"] == 10
    assert results["avg_nll"] >= 0.0
    assert math.isclose(results["perplexity"], math.exp(results["avg_nll"]), rel_tol=1e-6)


def test_dataset_formatters_build_piqa_and_hellaswag_examples():
    def fake_loader(name, config=None, split=None):
        if name == "piqa":
            return [
                {"goal": "open a jar", "sol1": "use a spoon", "sol2": "use a wrench", "label": 0, "id": "p1"},
            ]
        assert name == "hellaswag"
        return [
            {
                "activity_label": "Cooking",
                "ctx": "someone chops onions",
                "endings": [" they fry them", " they drive away"],
                "label": "0",
                "ind": "h1",
            }
        ]

    piqa = load_piqa_examples(dataset_loader=fake_loader)
    hellaswag = load_hellaswag_examples(dataset_loader=fake_loader)

    assert piqa == [
        MultipleChoiceExample(
            prompt="Question: open a jar\nAnswer:",
            choices=(" use a spoon", " use a wrench"),
            label=0,
            source_id="p1",
        )
    ]
    assert hellaswag == [
        MultipleChoiceExample(
            prompt="Context: Cooking: someone chops onions\nEnding:",
            choices=(" they fry them", " they drive away"),
            label=0,
            source_id="h1",
        )
    ]


def test_load_piqa_examples_falls_back_when_datasets_rejects_legacy_script(monkeypatch):
    def fake_loader(name, config=None, split=None):
        raise RuntimeError("Dataset scripts are no longer supported, but found piqa.py")

    expected = [
        MultipleChoiceExample(
            prompt="Question: fallback\nAnswer:",
            choices=(" a", " b"),
            label=1,
            source_id="0",
        )
    ]

    monkeypatch.setattr(
        "mathllm.evaluation.zero_forgetting._load_piqa_examples_from_original_files",
        lambda limit, split="validation", cache_dir=None: expected,
    )

    assert load_piqa_examples(dataset_loader=fake_loader) == expected


def test_run_zero_forgetting_benchmark_aggregates_metrics_and_renders_table(monkeypatch):
    monkeypatch.setattr(
        "mathllm.evaluation.zero_forgetting.compute_perplexity",
        lambda *args, **kwargs: {"perplexity": 10.0 if args[0] == "base" else 10.02, "avg_nll": 2.3, "num_tokens": 123},
    )
    monkeypatch.setattr(
        "mathllm.evaluation.zero_forgetting.benchmark_multiple_choice",
        lambda model, *args, **kwargs: {
            "accuracy": 0.75 if model == "base" else 0.751,
            "length_normalized_accuracy": 0.74 if model == "base" else 0.741,
            "num_examples": 5,
        },
    )

    results = run_zero_forgetting_benchmark(
        base_model="base",
        base_tokenizer="tok",
        arb_model="arb",
        arb_tokenizer="tok",
        device=torch.device("cpu"),
        wikitext_texts=["one", "two"],
        piqa_examples=[MultipleChoiceExample(prompt="p", choices=(" a", " b"), label=0)],
        hellaswag_examples=[MultipleChoiceExample(prompt="h", choices=(" a", " b"), label=1)],
    )

    assert set(results["benchmarks"]) == {"wikitext_103", "piqa", "hellaswag"}
    assert "WikiText-103" in results["markdown_table"]
    assert "PIQA" in results["markdown_table"]
    assert "HellaSwag" in results["markdown_table"]
    assert render_markdown_table(results) == results["markdown_table"]
