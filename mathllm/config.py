"""Configuration system using dataclasses with YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    injector_init_std: float = 1e-3  # small non-zero init so gradients reach extraction
    gate_init_logit: float = -2.0  # sigmoid(-2) ~ 0.12; learnable injection gate start
    extraction_mlp_hidden: int = 128  # hidden dim for extraction MLP
    extraction_num_classes: int = 10  # classes per digit (base-10)
    extraction_use_attention: bool = False  # use cross-position attention for extraction
    extraction_attn_rank: int = 32  # low-rank dimension for extraction attention


@dataclass
class DataConfig:
    num_positive: int = 50000
    num_negative: int = 50000
    num_edge_cases: int = 10000
    max_digits: int = 10
    max_value: int = 1_000_000_000
    seed: int = 42
    output_dir: str = "data/"
    pure_arithmetic: bool = False  # Generate only "A op B = C" format


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
    gradient_accumulation_steps: int = 1
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_dir: str = "checkpoints/"
    final_model_dir: str = "trained_model/"
    auto_resume_latest: bool = True
    device: str = "auto"
    answer_only_loss: bool = False  # full next-token loss for richer gradients
    early_stopping_patience: int = 3  # stop if eval loss doesn't improve for N evals
    max_eval_batches: int = 0  # cap eval batches per evaluation (0 = no cap)
    eval_batch_size: int = 0  # eval DataLoader batch size (0 = 2x batch_size)
    # Phased training
    phase1_epochs: int = 3  # extraction-only training
    phase2_epochs: int = 4  # extraction + injection, aux loss active
    phase3_epochs: int = 3  # end-to-end, aux loss decayed
    aux_loss_weight: float = 1.0  # lambda for auxiliary extraction loss
    aux_loss_decay: float = 0.1  # multiplier for aux weight in phase 3
    phase1_aux_only: bool = True  # filter to aux-eligible examples in Phase 1
    phase1_aux_threshold: float = 0.05  # stay in phase 1 until aux_eval < this
    curriculum_schedule: tuple[tuple[float, int], ...] = ((0.0, 3), (0.4, 6), (0.7, 10))


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
                value = tuple(
                    tuple(item) if isinstance(item, list) else item
                    for item in value
                )
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


def save_config(config: Config, path: str | Path) -> None:
    """Persist a config dataclass tree as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
