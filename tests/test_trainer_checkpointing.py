"""Tests for resumable trainer checkpointing."""

from __future__ import annotations

import copy
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mathllm.config import TrainingConfig
from mathllm.training.trainer import (
    ARBTrainer,
    find_latest_checkpoint,
    resolve_resume_checkpoint,
)


class ToyDataset(Dataset):
    """Small deterministic dataset for trainer tests."""

    def __init__(self, values: list[int]):
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        value = self.values[idx]
        return {
            "input_ids": torch.tensor([value], dtype=torch.long),
            "attention_mask": torch.tensor([1], dtype=torch.long),
            "labels": torch.tensor([0], dtype=torch.long),
            "digits_a": torch.zeros(10, dtype=torch.float32),
            "digits_b": torch.zeros(10, dtype=torch.float32),
            "has_aux": torch.tensor(False),
        }


class _DummySubModule(nn.Module):
    """Placeholder submodule for extract/inject."""
    def __init__(self):
        super().__init__()


class DummyARB(nn.Module):
    """Minimal trainable block that exposes a state_dict."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.extract = _DummySubModule()
        self.inject = _DummySubModule()


class DummyModel(nn.Module):
    """Minimal trainer-compatible model."""

    def __init__(self):
        super().__init__()
        self.arbs = nn.ModuleDict({"layer_0": DummyARB()})

    def get_trainable_parameters(self):
        return list(self.arbs.parameters())

    def forward(self, input_ids, attention_mask=None, labels=None):
        scale = self.arbs["layer_0"].scale
        prediction = input_ids.float() * scale
        target = labels.float()
        mask = attention_mask.float()
        loss = (((prediction - target) ** 2) * mask).mean()
        return {"loss": loss, "arb_extractions": {}}


def make_config(checkpoint_dir: str) -> TrainingConfig:
    """Build a compact training config for tests."""
    return TrainingConfig(
        lr=0.1,
        weight_decay=0.0,
        batch_size=1,
        max_epochs=1,
        checkpoint_every_steps=1,
        max_seq_len=8,
        warmup_steps=0,
        grad_clip=10.0,
        log_every=1,
        eval_every=0,
        checkpoint_dir=checkpoint_dir,
        auto_resume_latest=True,
        device="cpu",
        early_stopping_patience=3,
        # Skip phased training for dummy model tests
        phase1_epochs=0,
        phase2_epochs=0,
        phase3_epochs=1,
    )


def make_loader() -> DataLoader:
    """Build the shared deterministic train loader."""
    return DataLoader(ToyDataset([1, 2, 3, 4]), batch_size=1, shuffle=True, num_workers=0)


class TestTrainerCheckpointing:
    def test_checkpoint_round_trip_restores_progress_and_rng(self, tmp_path):
        config = make_config(str(tmp_path))
        trainer = ARBTrainer(
            model=DummyModel(),
            config=config,
            train_loader=make_loader(),
            device=torch.device("cpu"),
        )

        trainer.train(steps_to_run=2)
        trainer.best_eval_loss = 1.25
        trainer.patience_counter = 2

        random.seed(1234)
        torch.manual_seed(5678)
        trainer._save_checkpoint("unit")

        expected_python = random.random()
        expected_torch = torch.rand(1)

        reloaded = ARBTrainer(
            model=DummyModel(),
            config=config,
            train_loader=make_loader(),
            device=torch.device("cpu"),
        )
        reloaded.load_checkpoint(tmp_path / "arb_unit.pt")

        assert reloaded.global_step == trainer.global_step
        assert reloaded.completed_epochs == trainer.completed_epochs
        assert reloaded.batches_completed_in_epoch == trainer.batches_completed_in_epoch
        assert reloaded.best_eval_loss == 1.25
        assert reloaded.patience_counter == 2
        assert (tmp_path / "arb_latest.pt").exists()
        assert (tmp_path / "arb_step_1.pt").exists()
        assert (tmp_path / "arb_step_2.pt").exists()

        trainer_state = trainer.optimizer.state_dict()
        reloaded_state = reloaded.optimizer.state_dict()
        trainer_param_id = next(iter(trainer_state["state"]))
        reloaded_param_id = next(iter(reloaded_state["state"]))
        torch.testing.assert_close(
            trainer_state["state"][trainer_param_id]["exp_avg"],
            reloaded_state["state"][reloaded_param_id]["exp_avg"],
        )
        torch.testing.assert_close(
            trainer_state["state"][trainer_param_id]["exp_avg_sq"],
            reloaded_state["state"][reloaded_param_id]["exp_avg_sq"],
        )
        torch.testing.assert_close(
            trainer.model.arbs["layer_0"].scale.detach(),
            reloaded.model.arbs["layer_0"].scale.detach(),
        )
        assert random.random() == expected_python
        torch.testing.assert_close(torch.rand(1), expected_torch)

    def test_resolve_resume_checkpoint_prefers_explicit_path(self, tmp_path):
        latest = tmp_path / "arb_latest.pt"
        explicit = tmp_path / "arb_manual.pt"
        latest.touch()
        explicit.touch()

        assert find_latest_checkpoint(tmp_path) == latest
        assert resolve_resume_checkpoint(tmp_path) == latest
        assert resolve_resume_checkpoint(tmp_path, explicit_resume=explicit) == explicit
        assert resolve_resume_checkpoint(tmp_path, disable_resume=True) is None

    def test_chunked_training_matches_continuous_training(self, tmp_path):
        state = copy.deepcopy(DummyModel().state_dict())

        continuous_model = DummyModel()
        continuous_model.load_state_dict(state)
        continuous_trainer = ARBTrainer(
            model=continuous_model,
            config=make_config(str(tmp_path / "continuous")),
            train_loader=make_loader(),
            device=torch.device("cpu"),
        )
        continuous_trainer.train(steps_to_run=4)

        chunked_model = DummyModel()
        chunked_model.load_state_dict(state)
        chunked_config = make_config(str(tmp_path / "chunked"))
        chunked_trainer = ARBTrainer(
            model=chunked_model,
            config=chunked_config,
            train_loader=make_loader(),
            device=torch.device("cpu"),
        )
        chunked_trainer.train(steps_to_run=2)

        resumed_model = DummyModel()
        resumed_trainer = ARBTrainer(
            model=resumed_model,
            config=chunked_config,
            train_loader=make_loader(),
            device=torch.device("cpu"),
        )
        resumed_trainer.load_checkpoint(find_latest_checkpoint(chunked_config.checkpoint_dir))
        resumed_trainer.train(steps_to_run=2)

        assert continuous_trainer.global_step == 4
        assert resumed_trainer.global_step == 4
        torch.testing.assert_close(
            continuous_model.arbs["layer_0"].scale.detach(),
            resumed_model.arbs["layer_0"].scale.detach(),
        )
