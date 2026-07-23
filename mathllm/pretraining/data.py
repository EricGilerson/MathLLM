"""Deterministic prose/arithmetic mixture construction for toy pretraining."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from mathllm.pretraining.arithmetic_bpe_tokenizer import ArithmeticBPETokenizer


@dataclass(frozen=True)
class MixtureSpec:
    context_length: int
    train_blocks: int
    eval_blocks: int
    arithmetic_token_fraction: float
    max_digits: int
    invocation_fraction: float
    seed: int


def load_prose_documents(limit: int, fallback_path: str = "data/retention.txt") -> list[str]:
    """Load cached WikiText prose, falling back to the repository retention text."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        documents = [row["text"].strip() for row in dataset if row["text"].strip()]
        if documents:
            return documents[:limit]
    except Exception:
        pass

    path = Path(fallback_path)
    if not path.exists():
        raise FileNotFoundError("No cached WikiText data or fallback prose file is available")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()][:limit]


def _sample_expression(rng: random.Random, max_digits: int) -> tuple[int, str, int, int]:
    upper = 10**max_digits - 1
    op = rng.choice(["+", "-", "*", "/"])
    if op == "/":
        divisor = rng.randint(1, upper)
        quotient = rng.randint(1, upper)
        return divisor * quotient, op, divisor, quotient
    a, b = rng.randint(0, upper), rng.randint(0, upper)
    if op == "-" and a < b:
        a, b = b, a
    result = a + b if op == "+" else a - b if op == "-" else a * b
    return a, op, b, result


def arithmetic_texts(count: int, seed: int, max_digits: int, invocation_fraction: float) -> list[str]:
    rng = random.Random(seed)
    words = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}
    texts = []
    for _ in range(count):
        a, op, b, result = _sample_expression(rng, max_digits)
        equation = f"{a}{op}{b}={result}"
        if rng.random() < invocation_fraction:
            texts.append(f"Compute {a} {words[op]} {b}. Equation: {equation}\n")
        else:
            texts.append(equation + "\n")
    return texts


def _blocks_from_texts(texts: list[str], tokenizer: ArithmeticBPETokenizer, block_length: int) -> list[list[int]]:
    stream = []
    for text in texts:
        stream.extend(tokenizer.encode(text))
    if len(stream) < block_length + 1:
        raise ValueError("Source text is too small to form one training block")
    blocks = []
    for start in range(0, len(stream) - block_length, block_length):
        blocks.append(stream[start:start + block_length + 1])
    return blocks


def build_mixture(spec: MixtureSpec, prose_documents: list[str], tokenizer: ArithmeticBPETokenizer) -> dict[str, object]:
    """Build fixed-length blocks whose source mix is exact by block/token count."""
    if not 0.0 < spec.arithmetic_token_fraction < 1.0:
        raise ValueError("arithmetic_token_fraction must be between 0 and 1")
    rng = random.Random(spec.seed)
    train_arithmetic_blocks = round(spec.train_blocks * spec.arithmetic_token_fraction)
    eval_arithmetic_blocks = round(spec.eval_blocks * spec.arithmetic_token_fraction)
    train_prose_blocks = spec.train_blocks - train_arithmetic_blocks
    eval_prose_blocks = spec.eval_blocks - eval_arithmetic_blocks
    needed_texts = max((spec.train_blocks + spec.eval_blocks) * 3, 256)
    # Keep held-out sources genuinely disjoint rather than merely selecting
    # different blocks from one long token stream.
    if len(prose_documents) < 2:
        # The repository fallback can be a single long document. This keeps a
        # smoke test usable; real runs use many WikiText documents and are
        # disjoint by document below.
        train_prose = [document + "\n" for document in prose_documents]
        eval_prose = list(train_prose)
        prose_split = "shared fallback document only"
    else:
        split = max(1, int(len(prose_documents) * 0.9))
        if split == len(prose_documents):
            split -= 1
        train_prose = [document + "\n" for document in prose_documents[:split]]
        eval_prose = [document + "\n" for document in prose_documents[split:]]
        prose_split = "disjoint prose documents"
    train_arithmetic = arithmetic_texts(needed_texts, spec.seed + 1, spec.max_digits, spec.invocation_fraction)
    eval_arithmetic = arithmetic_texts(needed_texts, spec.seed + 2, spec.max_digits, spec.invocation_fraction)
    block_length = spec.context_length
    train_prose_pool = _blocks_from_texts(train_prose, tokenizer, block_length)
    eval_prose_pool = _blocks_from_texts(eval_prose, tokenizer, block_length)
    train_arithmetic_pool = _blocks_from_texts(train_arithmetic, tokenizer, block_length)
    eval_arithmetic_pool = _blocks_from_texts(eval_arithmetic, tokenizer, block_length)

    def take(pool, count):
        return [pool[index % len(pool)] for index in range(count)]

    train_records = [(block, 0) for block in take(train_prose_pool, train_prose_blocks)]
    train_records += [(block, 1) for block in take(train_arithmetic_pool, train_arithmetic_blocks)]
    eval_records = [(block, 0) for block in take(eval_prose_pool, eval_prose_blocks)]
    eval_records += [(block, 1) for block in take(eval_arithmetic_pool, eval_arithmetic_blocks)]
    rng.shuffle(train_records)
    rng.shuffle(eval_records)
    return {
        "train_input_ids": torch.tensor([record[0] for record in train_records], dtype=torch.long),
        "train_sources": torch.tensor([record[1] for record in train_records], dtype=torch.long),
        "eval_input_ids": torch.tensor([record[0] for record in eval_records], dtype=torch.long),
        "eval_sources": torch.tensor([record[1] for record in eval_records], dtype=torch.long),
        "metadata": {
            "context_length": spec.context_length,
            "train_blocks": spec.train_blocks,
            "eval_blocks": spec.eval_blocks,
            "arithmetic_token_fraction_target": spec.arithmetic_token_fraction,
            "train_arithmetic_blocks": train_arithmetic_blocks,
            "train_prose_blocks": train_prose_blocks,
            "eval_arithmetic_blocks": eval_arithmetic_blocks,
            "eval_prose_blocks": eval_prose_blocks,
            "source_split": f"{prose_split}; independent arithmetic seeds",
            "seed": spec.seed,
        },
    }


def save_mixture(path: str | Path, mixture: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mixture, path)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(mixture["metadata"], indent=2) + "\n")
