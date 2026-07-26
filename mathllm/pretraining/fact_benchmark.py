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


def _alpha_identifier(index: int) -> str:
    """Fixed-width alphabetic code with no digits for opaque fact tokens."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    value = index
    chars = []
    for _ in range(5):
        chars.append(alphabet[value % len(alphabet)])
        value //= len(alphabet)
    if value:
        raise ValueError("atomic fact experiment supports at most 26**5 bindings")
    return "".join(reversed(chars))


def make_atomic_facts(count: int, seed: int) -> list[Fact]:
    """Create arbitrary one-token entity/value bindings.

    The spellings are deliberately alphabetic and semantically empty.  They
    become tokenizer special tokens, rather than BPE fragments, so evaluation
    is a single next-token classification decision.
    """
    if count <= 0:
        raise ValueError("fact count must be positive")
    rng = random.Random(seed)
    values = [f"<factvalue{_alpha_identifier(index)}>" for index in range(count)]
    rng.shuffle(values)
    return [
        Fact(entity=f"<factentity{_alpha_identifier(index)}>", value=values[index])
        for index in range(count)
    ]


def atomic_fact_special_tokens(facts: list[Fact]) -> list[str]:
    """All opaque symbols needed by the atomic retrieval protocol."""
    return ["<factmap>"] + [fact.entity for fact in facts] + [fact.value for fact in facts]


def atomic_fact_training_texts(count: int, seed: int, facts: list[Fact]) -> list[str]:
    """Shuffled association passes, packed later into fixed-length blocks.

    A fact appears once per full pass through the mapping.  With normal block
    sizes this prevents an answer position from seeing a duplicate of its own
    binding earlier in the same context window, ruling out a local copy route.
    """
    rng = random.Random(seed)
    texts = []
    while len(texts) < count:
        order = list(facts)
        rng.shuffle(order)
        texts.extend(f"{fact.entity}<factmap>{fact.value}\n" for fact in order)
    return texts[:count]


def atomic_fact_eval_cases(facts: list[Fact]) -> list[tuple[str, str]]:
    """One next-token query per stored binding; no answer appears in its prompt."""
    return [(f"{fact.entity}<factmap>", fact.value) for fact in facts]


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
    """Build a held-out-template evaluation corpus without duplicate records.

    A small fact vocabulary can otherwise provide only one held-out record per
    binding, which is insufficient to form the configured number of *unique*
    fixed-length evaluation blocks.  Each fact therefore appears once under
    each disjoint held-out wording.  These records never use a training query
    template; they are only for the held-out loss source, not fact accuracy.
    """
    rng = random.Random(seed)
    records = [
        template.format(entity=fact.entity, value=fact.value) + "\n"
        for fact in facts
        for template in _EVAL_Q
    ]
    rng.shuffle(records)
    return records[:min(count, len(records))]
