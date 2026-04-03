"""Training loop for ARB parameters.

Only the ARB learned parameters (extraction and injection projections) are
trained. The base model and frozen ARB stages never receive gradients.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mathllm.config import TrainingConfig
from mathllm.model.gpt2_arb import GPT2WithARB

logger = logging.getLogger(__name__)


class ARBTrainer:
    """Train only the ARB learned interface on arithmetic data."""

    def __init__(
        self,
        model: GPT2WithARB,
        config: TrainingConfig,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device or torch.device("cpu")

        # Collect ONLY ARB learned parameters
        self.arb_params = model.get_trainable_parameters()
        param_count = sum(p.numel() for p in self.arb_params)
        logger.info(f"Training {param_count:,} ARB parameters")

        self.optimizer = torch.optim.AdamW(
            self.arb_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Linear warmup then linear decay schedule
        total_steps = len(train_loader) * config.max_epochs
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_schedule(config.warmup_steps, total_steps),
        )

        self.global_step = 0
        self.best_eval_loss = float("inf")

    @staticmethod
    def _lr_schedule(warmup_steps: int, total_steps: int):
        """Linear warmup then linear decay."""
        def schedule(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 1.0 - progress)
        return schedule

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            Dictionary of training metrics history.
        """
        self.model.to(self.device)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }

        for epoch in range(self.config.max_epochs):
            epoch_loss = self._train_epoch(epoch, history)
            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs} — avg loss: {epoch_loss:.4f}")

            if self.eval_loader is not None:
                eval_loss = self._evaluate()
                history["eval_loss"].append(eval_loss)
                logger.info(f"  Eval loss: {eval_loss:.4f}")

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint("best")

            self._save_checkpoint(f"epoch_{epoch + 1}")

        return history

    def _train_epoch(self, epoch: int, history: dict[str, list[float]]) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.arb_params, self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every == 0:
                avg = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"  Step {self.global_step}: loss={avg:.4f} lr={lr:.2e}"
                )
                history["train_loss"].append(avg)
                history["learning_rate"].append(lr)

            if (
                self.eval_loader is not None
                and self.config.eval_every > 0
                and self.global_step % self.config.eval_every == 0
            ):
                eval_loss = self._evaluate()
                history["eval_loss"].append(eval_loss)
                logger.info(f"  Step {self.global_step}: eval_loss={eval_loss:.4f}")
                self.model.train()

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate on the eval set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs["loss"].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, name: str) -> None:
        """Save ARB parameters only (not the frozen base model)."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"arb_{name}.pt"

        # Save only ARB state dicts
        arb_state = {}
        for key, arb in self.model.arbs.items():
            arb_state[key] = arb.state_dict()

        torch.save(
            {
                "arb_state": arb_state,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_eval_loss": self.best_eval_loss,
            },
            path,
        )
        logger.info(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load ARB parameters from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for key, state in ckpt["arb_state"].items():
            self.model.arbs[key].load_state_dict(state)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["global_step"]
        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")
