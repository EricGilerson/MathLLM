"""Deterministic parametric fact-retrieval data for capacity experiments.

Facts and questions are generated locally so the benchmark has no web-data
leakage, licensing dependency, or accidental answer overlap.  A query record
never contains its supporting statement: answering it requires the model to
store the entity--attribute binding in its parameters during training.
"""
from __future__ import annotations

from dataclasses import dataclass
import random


_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z")
_VOWELS = ("a", "e", "i", "o", "u")
_CODAS = ("l", "m", "n", "r", "s", "th", "v")
_TRAIN_Q = (
    "Question: What vocation does {entity} have? Answer: {value}.",
    "Question: What is {entity}'s vocation? Answer: {value}.",
    "Question: Which vocation belongs to {entity}? Answer: {value}.",
)
_EVAL_Q = (
    "Question: Identify {entity}'s occupation. Answer: {value}.",
    "Question: What work is associated with {entity}? Answer: {value}.",
    "Question: {entity} works as what? Answer: {value}.",
)


@dataclass(frozen=True)
class Fact:
    entity: str
    value: str


def _word(index: int, prefix: str) -> str:
    """Injective, pronounceable synthetic identifier; no semantic prior."""
    base = len(_ONSETS) * len(_VOWELS) * len(_CODAS)
    chunks = []
    value = index
    while not chunks or value:
        digit = value % base
        value //= base
        onset = _ONSETS[digit % len(_ONSETS)]
        digit //= len(_ONSETS)
        vowel = _VOWELS[digit % len(_VOWELS)]
        digit //= len(_VOWELS)
        coda = _CODAS[digit % len(_CODAS)]
        chunks.append(onset + vowel + coda)
    return prefix + "".join(reversed(chunks))


def make_facts(count: int, seed: int) -> list[Fact]:
    if count <= 0:
        raise ValueError("fact count must be positive")
    rng = random.Random(seed)
    values = [_word(index, "v") for index in range(max(64, count))]
    rng.shuffle(values)
    return [Fact(entity=_word(index, "e"), value=values[index]) for index in range(count)]


def fact_training_texts(count: int, seed: int, facts: list[Fact]) -> list[str]:
    """Statements and answer-bearing queries are separate records.

    This prevents the trivial attention-copy solution of placing a fact beside
    its answer in one context window.
    """
    rng = random.Random(seed)
    texts = []
    for _ in range(count):
        fact = rng.choice(facts)
        if rng.random() < 0.5:
            texts.append(f"{fact.entity} has vocation {fact.value}.\n")
        else:
            texts.append(rng.choice(_TRAIN_Q).format(entity=fact.entity, value=fact.value) + "\n")
    return texts


def fact_eval_cases(count: int, seed: int, facts: list[Fact]) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    selected = [facts[index % len(facts)] for index in rng.sample(range(len(facts)), min(count, len(facts)))]
    return [
        (rng.choice(_EVAL_Q).format(entity=fact.entity, value="")[:-1], fact.value)
        for fact in selected
    ]


def fact_seen_template_cases(count: int, seed: int, facts: list[Fact]) -> list[tuple[str, str]]:
    """Same binding probe, but with question wording seen during training."""
    rng = random.Random(seed)
    selected = [facts[index % len(facts)] for index in rng.sample(range(len(facts)), min(count, len(facts)))]
    return [
        (rng.choice(_TRAIN_Q).format(entity=fact.entity, value="")[:-1], fact.value)
        for fact in selected
    ]


def fact_eval_texts(count: int, seed: int, facts: list[Fact]) -> list[str]:
    return [f"{prompt} {answer}.\n" for prompt, answer in fact_eval_cases(count, seed, facts)]
