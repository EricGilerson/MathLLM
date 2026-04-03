"""Training loop for ARB parameters.

Only the ARB learned parameters (extraction and injection projections) are
trained. The base model and frozen ARB stages never receive gradients.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

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
        self.patience_counter = 0
        self.early_stopped = False

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

        epoch_bar = tqdm(
            range(self.config.max_epochs),
            desc="Training",
            unit="epoch",
            position=0,
            leave=True,
        )

        for epoch in epoch_bar:
            epoch_loss = self._train_epoch(epoch, history)
            epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}")

            if self.eval_loader is not None:
                eval_loss = self._evaluate()
                history["eval_loss"].append(eval_loss)
                epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}", eval=f"{eval_loss:.4f}")

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    tqdm.write(
                        f"Early stopping at epoch {epoch + 1} — "
                        f"no improvement for {self.config.early_stopping_patience} evals"
                    )
                    self.early_stopped = True
                    break

            self._save_checkpoint(f"epoch_{epoch + 1}")

        epoch_bar.close()
        return history

    def _train_epoch(self, epoch: int, history: dict[str, list[float]]) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        batch_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            position=1,
            leave=False,
        )

        import contextlib
        # MPS supports float16 natively
        use_autocast = self.device.type == "mps" or self.device.type == "cuda"
        dtype = torch.float16 if self.device.type == "mps" else torch.bfloat16
        autocast_ctx = torch.autocast(device_type=self.device.type, dtype=dtype) if use_autocast else contextlib.nullcontext()

        for batch in batch_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast_ctx:
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
            # set_to_none=True avoids zeroing out memory, which saves bandwidth
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.device.type == "mps":
                torch.mps.synchronize()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar with running metrics
            avg = total_loss / num_batches
            lr = self.scheduler.get_last_lr()[0]
            batch_bar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}", step=self.global_step)

            if self.global_step % self.config.log_every == 0:
                history["train_loss"].append(avg)
                history["learning_rate"].append(lr)

            if (
                self.eval_loader is not None
                and self.config.eval_every > 0
                and self.global_step % self.config.eval_every == 0
            ):
                eval_loss = self._evaluate()
                history["eval_loss"].append(eval_loss)
                batch_bar.set_postfix(
                    loss=f"{avg:.4f}", lr=f"{lr:.2e}", eval=f"{eval_loss:.4f}"
                )
                self.model.train()

        batch_bar.close()
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate on the eval set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        eval_bar = tqdm(
            self.eval_loader,
            desc="Evaluating",
            unit="batch",
            position=2,
            leave=False,
        )

        import contextlib
        use_autocast = self.device.type == "mps" or self.device.type == "cuda"
        dtype = torch.float16 if self.device.type == "mps" else torch.bfloat16
        autocast_ctx = torch.autocast(device_type=self.device.type, dtype=dtype) if use_autocast else contextlib.nullcontext()

        for batch in eval_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast_ctx:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            total_loss += outputs["loss"].item()
            num_batches += 1
            eval_bar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        eval_bar.close()
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
