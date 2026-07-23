#!/usr/bin/env python3
"""Measure ARB trigger rates and NLL deltas on a small curated probe suite.

This is intentionally a reproducible diagnostic probe, not a replacement for
held-out WikiText or a held-out code corpus. It reports exactly which examples
are affected by a valid arithmetic trigger.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.evaluation.zero_forgetting import compute_perplexity


DEFAULT_CASES = {
    "prose": [
        "The committee met on Tuesday to discuss the proposal.",
        "A quiet rain fell over the empty street after sunset.",
        "Please return the library book before the end of the month.",
        "The experiment was repeated after the temperature stabilized.",
    ],
    "code": [
        "total = price + tax\nprint(total)",
        "if status == 200:\n    return response.json()",
        "count += 1\nitems.append(count)",
        "def scale(x):\n    return x * 2",
    ],
    "identifiers": [
        "Invoice = Q3-2026-041; the due date is 2026-08-15.",
        "The room code A=17 is printed on the visitor badge.",
        "The table lists x = y as a symbolic equality, not a calculation.",
        "Call +1 212 555 0184 after 09:30.",
    ],
    "quoted_or_wrong": [
        "The student wrote, '2+2=5,' in the margin.",
        "In the example, the text literally says '7/2=3.5'.",
        "The document quotes the line '3-5=-2' without evaluating it.",
        "The log records: 8*9=71, which was later corrected.",
    ],
    "supported_equation": [
        "84/7=", "347 * 291 =", "482+319=", "999-123=",
    ],
}


def nll(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    if input_ids.size(1) < 2:
        return 0.0
    return float(F.cross_entropy(logits[:, :-1].transpose(1, 2), input_ids[:, 1:]).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="trained_model_360m")
    parser.add_argument("--output", default="review/artifacts/containment_probe.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wikitext-limit", type=int, default=0,
                        help="Add this many non-empty WikiText-103 test documents from the local cache.")
    parser.add_argument("--swebench-limit", type=int, default=0,
                        help="Add this many SWE-bench Lite test patches from the local cache.")
    parser.add_argument("--swebench-offset", type=int, default=0,
                        help="Starting SWE-bench Lite test-patch index.")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Score at most this many tokens from each example.")
    parser.add_argument("--wikitext-ppl-limit", type=int, default=0,
                        help="Compute base/ARB perplexity over this many cached WikiText test documents.")
    parser.add_argument("--wikitext-offset", type=int, default=0,
                        help="Starting non-empty WikiText test-document index for corpus or PPL slices.")
    parser.add_argument("--ppl-window", type=int, default=256)
    parser.add_argument("--ppl-max-document-tokens", type=int, default=0,
                        help="If positive, truncate each WikiText document before PPL scoring.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer, _ = GPT2WithARB.from_exported_model(args.model_dir, device=device)
    model.eval()
    model.base_model.eval()

    cases = {category: list(texts) for category, texts in DEFAULT_CASES.items()}
    if args.wikitext_limit or args.swebench_limit or args.wikitext_ppl_limit:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from datasets import load_dataset
        if args.wikitext_limit or args.wikitext_ppl_limit:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
            wikitext_texts = [
                row["text"] for row in dataset if row["text"].strip()
            ]
            if args.wikitext_limit:
                cases["wikitext_103_test"] = wikitext_texts[
                    args.wikitext_offset:args.wikitext_offset + args.wikitext_limit
                ]
        if args.swebench_limit:
            dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
            cases["swebench_lite_patches"] = [
                row["patch"] for row in dataset if row["patch"].strip()
            ][args.swebench_offset:args.swebench_offset + args.swebench_limit]

    rows = []
    for category, texts in cases.items():
        for text in texts:
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"][:, :args.max_tokens].to(device)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                arb = model(input_ids=input_ids, attention_mask=attention_mask)
                base_logits = model.base_model(
                    input_ids=input_ids, attention_mask=attention_mask,
                ).logits
                first_injector = next(iter(model.injectors.values()))
                inject_mask = first_injector._build_smart_inject_mask(
                    input_ids, arb["arb_detection"]["eq_pos"],
                    arb["arb_detection"]["has_valid_equation"],
                )
            detection = arb["arb_detection"]
            rows.append({
                "category": category,
                "text": text,
                "token_count": int(input_ids.numel()),
                "candidate_equals": int(detection["candidate_eq_count"][0]),
                "syntax_valid": bool(detection["syntax_valid"][0]),
                "domain_valid": bool(detection["domain_valid"][0]),
                "arb_active_tokens": int(inject_mask.sum().item()),
                "base_nll": nll(base_logits, input_ids),
                "arb_nll": nll(arb["logits"], input_ids),
            })
    for row in rows:
        row["nll_delta"] = row["arb_nll"] - row["base_nll"]

    summaries = {}
    for category in cases:
        category_rows = [row for row in rows if row["category"] == category]
        summaries[category] = {
            "examples": len(category_rows),
            "equals_candidates": sum(row["candidate_equals"] for row in category_rows),
            "valid_triggers": sum(row["domain_valid"] for row in category_rows),
            "affected_tokens": sum(row["arb_active_tokens"] for row in category_rows),
            "mean_base_nll": sum(row["base_nll"] for row in category_rows) / len(category_rows),
            "mean_arb_nll": sum(row["arb_nll"] for row in category_rows) / len(category_rows),
            "mean_nll_delta": sum(row["nll_delta"] for row in category_rows) / len(category_rows),
        }
    perplexity = None
    if args.wikitext_ppl_limit:
        ppl_texts = wikitext_texts[
            args.wikitext_offset:args.wikitext_offset + args.wikitext_ppl_limit
        ]
        if args.ppl_max_document_tokens:
            ppl_texts = [
                tokenizer.decode(
                    tokenizer.encode(text, add_special_tokens=False)[:args.ppl_max_document_tokens],
                    skip_special_tokens=True,
                )
                for text in ppl_texts
            ]
        base_ppl = compute_perplexity(
            model.base_model, tokenizer, ppl_texts, device=device,
            max_length=args.ppl_window, stride=args.ppl_window,
        )
        arb_ppl = compute_perplexity(
            model, tokenizer, ppl_texts, device=device,
            max_length=args.ppl_window, stride=args.ppl_window,
        )
        perplexity = {
            "corpus": "WikiText-103 test", "documents": len(ppl_texts),
            "document_offset": args.wikitext_offset,
            "max_length": args.ppl_window, "stride": args.ppl_window,
            "max_document_tokens": args.ppl_max_document_tokens or None,
            "base": base_ppl, "arb": arb_ppl,
            "perplexity_delta": arb_ppl["perplexity"] - base_ppl["perplexity"],
            "avg_nll_delta": arb_ppl["avg_nll"] - base_ppl["avg_nll"],
        }
    report = {
        "date": str(date.today()), "model_dir": args.model_dir, "device": str(device),
        "probe": "curated plus optional offline corpora", "max_tokens": args.max_tokens,
        "summary": summaries, "perplexity": perplexity, "examples": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "summary": summaries, "perplexity": perplexity}, indent=2))


if __name__ == "__main__":
    main()
