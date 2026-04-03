"""Training loop for ARB parameters.

Only the ARB learned parameters (extraction and injection projections) are
trained. The base model and frozen ARB stages never receive gradients.
"""

from __future__ import annotations

import contextlib
import logging
import random
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from mathllm.config import TrainingConfig
from mathllm.model.gpt2_arb import GPT2WithARB

logger = logging.getLogger(__name__)

_TRAIN_SAMPLER_SEED = 17


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the default checkpoint to resume from, if one exists."""
    ckpt_dir = Path(checkpoint_dir)
    latest = ckpt_dir / "arb_latest.pt"
    if latest.exists():
        return latest

    candidates = sorted(
        ckpt_dir.glob("arb_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_resume_checkpoint(
    checkpoint_dir: str | Path,
    explicit_resume: str | Path | None = None,
    auto_resume_latest: bool = True,
    disable_resume: bool = False,
) -> Path | None:
    """Resolve checkpoint selection with explicit resume taking precedence."""
    if explicit_resume is not None:
        return Path(explicit_resume)
    if disable_resume or not auto_resume_latest:
        return None
    return find_latest_checkpoint(checkpoint_dir)


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

        self.steps_per_epoch = len(train_loader)
        self.total_training_steps = self.steps_per_epoch * config.max_epochs

        # Collect ONLY ARB learned parameters
        self.arb_params = model.get_trainable_parameters()
        param_count = sum(p.numel() for p in self.arb_params)
        logger.info(f"Training {param_count:,} ARB parameters")

        self.optimizer = torch.optim.AdamW(
            self.arb_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.global_step = 0
        self.completed_epochs = 0
        self.batches_completed_in_epoch = 0
        self.best_eval_loss = float("inf")
        self.patience_counter = 0
        self.early_stopped = False

    def _lr_scale(self, step: int) -> float:
        """Linear warmup then linear decay using 1-based training steps."""
        if self.total_training_steps <= 0:
            return 1.0

        if self.config.warmup_steps > 0 and step <= self.config.warmup_steps:
            return step / self.config.warmup_steps

        progress = (step - self.config.warmup_steps) / max(
            self.total_training_steps - self.config.warmup_steps,
            1,
        )
        return max(0.0, 1.0 - progress)

    def _set_learning_rate(self, step: int) -> float:
        """Update optimizer LR for the next optimizer step."""
        lr = self.config.lr * self._lr_scale(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _autocast_context(self):
        """Return an autocast context when the device supports it."""
        use_autocast = self.device.type in {"mps", "cuda"}
        if not use_autocast:
            return contextlib.nullcontext()

        dtype = torch.float16 if self.device.type == "mps" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _clone_loader(
        self,
        loader: DataLoader,
        *,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> DataLoader:
        """Clone a DataLoader with deterministic sampling settings."""
        kwargs = {
            "dataset": loader.dataset,
            "batch_size": loader.batch_size,
            "shuffle": shuffle,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "multiprocessing_context": loader.multiprocessing_context,
            "generator": generator,
            "persistent_workers": loader.persistent_workers,
            "pin_memory_device": loader.pin_memory_device,
        }
        if loader.num_workers > 0:
            kwargs["prefetch_factor"] = loader.prefetch_factor
        return DataLoader(**kwargs)

    def _build_epoch_train_loader(self, epoch: int) -> DataLoader:
        """Build a deterministic train loader for the requested epoch."""
        sampler = self.train_loader.sampler
        if isinstance(sampler, RandomSampler):
            generator = torch.Generator()
            generator.manual_seed(_TRAIN_SAMPLER_SEED + epoch)
            return self._clone_loader(self.train_loader, shuffle=True, generator=generator)
        if isinstance(sampler, SequentialSampler):
            return self._clone_loader(self.train_loader, shuffle=False)

        logger.warning(
            "Custom train sampler detected; resuming mid-epoch may not preserve batch order."
        )
        return self.train_loader

    def _capture_rng_state(self) -> dict[str, object]:
        """Capture RNG state needed for deterministic continuation."""
        state: dict[str, object] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict[str, object] | None) -> None:
        """Restore RNG state if the checkpoint contains it."""
        if not state:
            return
        python_state = state.get("python")
        if python_state is not None:
            random.setstate(python_state)
        torch_state = state.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        cuda_state = state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    def train(
        self,
        *,
        epochs_to_run: int | None = None,
        steps_to_run: int | None = None,
    ) -> dict[str, list[float]]:
        """Run training for the requested incremental budget."""
        if epochs_to_run is not None and epochs_to_run <= 0:
            raise ValueError("epochs_to_run must be positive")
        if steps_to_run is not None and steps_to_run <= 0:
            raise ValueError("steps_to_run must be positive")

        self.model.to(self.device)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }

        if self.global_step >= self.total_training_steps:
            logger.info(
                "Training horizon already reached (%s/%s steps).",
                self.global_step,
                self.total_training_steps,
            )
            return history

        epoch_limit = self.config.max_epochs
        if epochs_to_run is not None:
            epoch_limit = min(epoch_limit, self.completed_epochs + epochs_to_run)

        step_limit = self.total_training_steps
        if steps_to_run is not None:
            step_limit = min(step_limit, self.global_step + steps_to_run)

        made_progress = False
        epoch_bar = tqdm(
            range(self.completed_epochs, epoch_limit),
            desc="Training",
            unit="epoch",
            position=0,
            leave=True,
        )

        for epoch in epoch_bar:
            if self.global_step >= step_limit or self.global_step >= self.total_training_steps:
                break

            epoch_loss, epoch_completed = self._train_epoch(epoch, history, step_limit)
            if epoch_loss is not None:
                made_progress = True
                epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}")

            if not epoch_completed:
                break

            epoch_loss_value = epoch_loss if epoch_loss is not None else 0.0
            if self.eval_loader is not None:
                eval_loss = self._evaluate()
                history["eval_loss"].append(eval_loss)
                epoch_bar.set_postfix(loss=f"{epoch_loss_value:.4f}", eval=f"{eval_loss:.4f}")

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    tqdm.write(
                        f"Early stopping at epoch {epoch + 1} - "
                        f"no improvement for {self.config.early_stopping_patience} evals"
                    )
                    self.early_stopped = True

            self._save_checkpoint(f"epoch_{epoch + 1}")

            if self.early_stopped:
                break

        epoch_bar.close()

        if made_progress and self.batches_completed_in_epoch > 0:
            self._save_checkpoint(f"step_{self.global_step}")

        return history

    def _train_epoch(
        self,
        epoch: int,
        history: dict[str, list[float]],
        step_limit: int,
    ) -> tuple[float | None, bool]:
        """Train for part or all of one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        start_batch = self.batches_completed_in_epoch if epoch == self.completed_epochs else 0
        train_loader = self._build_epoch_train_loader(epoch)
        train_iter = iter(train_loader)

        for _ in range(start_batch):
            try:
                next(train_iter)
            except StopIteration:
                self.completed_epochs = epoch + 1
                self.batches_completed_in_epoch = 0
                return None, True

        batch_bar = tqdm(
            total=self.steps_per_epoch,
            initial=start_batch,
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            position=1,
            leave=False,
        )

        while (
            self.batches_completed_in_epoch < self.steps_per_epoch
            and self.global_step < step_limit
            and self.global_step < self.total_training_steps
        ):
            try:
                batch = next(train_iter)
            except StopIteration:
                break

            batch = {key: value.to(self.device) for key, value in batch.items()}

            with self._autocast_context():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"]

            lr = self._set_learning_rate(self.global_step + 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.arb_params, self.config.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.device.type == "mps":
                torch.mps.synchronize()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            self.batches_completed_in_epoch += 1

            avg = total_loss / num_batches
            batch_bar.update(1)
            batch_bar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}", step=self.global_step)

            if self.global_step % self.config.log_every == 0:
                history["train_loss"].append(avg)
                history["learning_rate"].append(lr)

            if (
                self.config.checkpoint_every_steps > 0
                and self.global_step % self.config.checkpoint_every_steps == 0
            ):
                self._save_checkpoint(f"step_{self.global_step}")

            if (
                self.eval_loader is not None
                and self.config.eval_every > 0
                and self.global_step % self.config.eval_every == 0
            ):
                eval_loss = self._evaluate()
                history["eval_loss"].append(eval_loss)
                batch_bar.set_postfix(
                    loss=f"{avg:.4f}",
                    lr=f"{lr:.2e}",
                    eval=f"{eval_loss:.4f}",
                    step=self.global_step,
                )
                self.model.train()

        batch_bar.close()

        epoch_completed = self.batches_completed_in_epoch >= self.steps_per_epoch
        if epoch_completed:
            self.completed_epochs = epoch + 1
            self.batches_completed_in_epoch = 0

        return total_loss / num_batches if num_batches else None, epoch_completed

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

        with self._autocast_context():
            for batch in eval_bar:
                batch = {key: value.to(self.device) for key, value in batch.items()}
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

    def _checkpoint_payload(self) -> dict[str, object]:
        """Build the full checkpoint payload."""
        arb_state = {key: arb.state_dict() for key, arb in self.model.arbs.items()}
        return {
            "arb_state": arb_state,
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "completed_epochs": self.completed_epochs,
            "batches_completed_in_epoch": self.batches_completed_in_epoch,
            "best_eval_loss": self.best_eval_loss,
            "patience_counter": self.patience_counter,
            "steps_per_epoch": self.steps_per_epoch,
            "max_epochs": self.config.max_epochs,
            "rng_state": self._capture_rng_state(),
        }

    def _save_checkpoint(self, name: str) -> None:
        """Save a resumable checkpoint and refresh the latest pointer."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload()

        path = ckpt_dir / f"arb_{name}.pt"
        torch.save(payload, path)
        latest_path = ckpt_dir / "arb_latest.pt"
        if latest_path != path:
            torch.save(payload, latest_path)

        logger.info(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a checkpoint and restore training progress."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for key, state in ckpt["arb_state"].items():
            self.model.arbs[key].load_state_dict(state)

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["global_step"]
        saved_steps_per_epoch = ckpt.get("steps_per_epoch", self.steps_per_epoch)

        if "completed_epochs" in ckpt:
            self.completed_epochs = ckpt["completed_epochs"]
            self.batches_completed_in_epoch = ckpt.get("batches_completed_in_epoch", 0)
        else:
            self.completed_epochs = self.global_step // max(saved_steps_per_epoch, 1)
            self.batches_completed_in_epoch = self.global_step % max(saved_steps_per_epoch, 1)

        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        self.patience_counter = ckpt.get("patience_counter", 0)
        self._restore_rng_state(ckpt.get("rng_state"))

        if saved_steps_per_epoch != self.steps_per_epoch:
            logger.warning(
                "Checkpoint was created with %s steps/epoch, current run has %s. "
                "Mid-epoch resume may be inaccurate if the dataset changed.",
                saved_steps_per_epoch,
                self.steps_per_epoch,
            )

        logger.info(
            "Loaded checkpoint from %s (step %s, epoch %s, batch %s)",
            path,
            self.global_step,
            self.completed_epochs,
            self.batches_completed_in_epoch,
        )
