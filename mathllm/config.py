"""Configuration system using dataclasses with YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RNSConfig:
    primes: tuple[int, ...] = (7, 11, 13, 17, 19, 23, 29, 31, 37)
    num_digit_slots: int = 10


@dataclass
class ARBConfig:
    layer_positions: tuple[int, ...] = (4, 8, 10)
    softmax_temperature: float = 1000.0
    num_results: int = 5  # add, sub, mul, exp, div
    dropout: float = 0.1  # dropout on extraction and injection layers


@dataclass
class DataConfig:
    num_positive: int = 50000
    num_negative: int = 50000
    num_edge_cases: int = 10000
    max_digits: int = 10
    max_value: int = 1_000_000_000
    seed: int = 42
    output_dir: str = "data/"


@dataclass
class TrainingConfig:
    base_model: str = "gpt2"
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_epochs: int = 10
    checkpoint_every_steps: int = 0
    max_seq_len: int = 128
    warmup_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_dir: str = "checkpoints/"
    auto_resume_latest: bool = True
    device: str = "auto"
    early_stopping_patience: int = 3  # stop if eval loss doesn't improve for N evals


@dataclass
class EvalConfig:
    num_samples_per_config: int = 200
    max_digits_range: tuple[int, int] = (1, 10)
    max_new_tokens: int = 20


@dataclass
class Config:
    rns: RNSConfig = field(default_factory=RNSConfig)
    arb: ARBConfig = field(default_factory=ARBConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


def _merge_into_dataclass(dc: Any, overrides: dict) -> Any:
    """Recursively merge a dict of overrides into a dataclass instance."""
    for key, value in overrides.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _merge_into_dataclass(current, value)
        else:
            if isinstance(value, list) and isinstance(current, tuple):
                value = tuple(value)
            setattr(dc, key, value)
    return dc


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML file, merging onto defaults."""
    config = Config()
    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                overrides = yaml.safe_load(f) or {}
            _merge_into_dataclass(config, overrides)
    return config
