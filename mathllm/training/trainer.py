"""Training loop for ARB parameters with phased training.

Phase 1: Train extraction only (freeze injection), auxiliary digit loss only.
Phase 2: Train extraction + injection, LM loss + auxiliary loss.
Phase 3: End-to-end, LM loss + decayed auxiliary loss.
"""

from __future__ import annotations

import contextlib
import logging
import random
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler

from mathllm.config import TrainingConfig
from mathllm.model.gpt2_arb import GPT2WithARB
from mathllm.training.losses import compute_extraction_loss

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

        self.accumulation_steps = max(config.gradient_accumulation_steps, 1)
        self._full_steps_per_epoch = len(train_loader) // self.accumulation_steps
        self.steps_per_epoch = self._full_steps_per_epoch
        self.total_training_steps = self.steps_per_epoch * config.max_epochs

        self.global_step = 0
        self.completed_epochs = 0
        self.batches_completed_in_epoch = 0
        self.best_eval_loss = float("inf")
        self.patience_counter = 0
        self.early_stopped = False
        self._current_phase = 0  # will be set on first epoch

        # Pre-compute aux-eligible indices for Phase 1 filtering
        dataset = train_loader.dataset
        if hasattr(dataset, "has_aux"):
            self._aux_indices = torch.where(dataset.has_aux)[0].tolist()
        else:
            self._aux_indices = None

        # Pre-compute max operand digits for curriculum filtering
        if hasattr(dataset, "max_operand_digits"):
            self._max_operand_digits = dataset.max_operand_digits
        else:
            self._max_operand_digits = None

        # Initial optimizer setup (will be rebuilt on phase transitions)
        self._build_optimizer()

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _get_phase(self, epoch: int) -> int:
        """Determine training phase from epoch number."""
        p1 = self.config.phase1_epochs
        p2 = p1 + self.config.phase2_epochs
        if epoch < p1:
            return 1
        if epoch < p2:
            return 2
        return 3

    def _configure_phase(self, phase: int) -> None:
        """Freeze/unfreeze parameters for the given phase and rebuild optimizer."""
        if phase == self._current_phase:
            return

        old_phase = self._current_phase
        self._current_phase = phase

        if phase == 1:
            # Extraction only: freeze injection, disable extraction dropout
            for arb in self.model.arbs.values():
                for p in arb.extract.parameters():
                    p.requires_grad = True
                for p in arb.inject.parameters():
                    p.requires_grad = False
                # Phase 1 is precision regression — dropout adds noise
                arb.extract.dropout.p = 0.0
        else:
            # Phase 2 & 3: everything trainable, restore dropout
            for arb in self.model.arbs.values():
                for p in arb.extract.parameters():
                    p.requires_grad = True
                for p in arb.inject.parameters():
                    p.requires_grad = True
                # Restore configured dropout for generalization
                arb.extract.dropout.p = self.model.config.arb.dropout

        # Reset early stopping on phase transitions — flat Phase 1 eval
        # should not count against Phase 2's patience budget.
        self.patience_counter = 0
        self.best_eval_loss = float("inf")

        # Adjust steps_per_epoch for filtered Phase 1 dataset
        if phase == 1 and self.config.phase1_aux_only and self._aux_indices:
            filtered_size = len(self._aux_indices)
            batch_size = self.train_loader.batch_size or 1
            self.steps_per_epoch = max(1, filtered_size // batch_size // self.accumulation_steps)
        else:
            self.steps_per_epoch = self._full_steps_per_epoch

        self._build_optimizer()
        logger.info(
            "Phase transition: %s -> %s (trainable params: %s)",
            old_phase, phase,
            sum(p.numel() for p in self.arb_params),
        )

    def _build_optimizer(self) -> None:
        """(Re)build optimizer with currently-trainable ARB parameters."""
        self.arb_params = self.model.get_trainable_parameters()
        if not self.arb_params:
            logger.warning("No trainable parameters found for optimizer")
            self.arb_params = [torch.zeros(1, requires_grad=True)]
        self.optimizer = torch.optim.AdamW(
            self.arb_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _compute_loss(
        self,
        outputs: dict,
        batch: dict[str, torch.Tensor],
        phase: int,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute total loss based on current training phase.

        Returns:
            (total_loss, lm_loss_value, aux_loss_value) for logging.
        """
        lm_loss = outputs["loss"]
        arb_extractions = outputs.get("arb_extractions", {})

        # Compute auxiliary extraction loss
        aux_loss = compute_extraction_loss(
            arb_extractions=arb_extractions,
            gt_digits_a=batch["digits_a"],
            gt_digits_b=batch["digits_b"],
            has_aux=batch["has_aux"],
            attention_mask=batch["attention_mask"],
            eq_positions=batch["eq_position"],
            op_positions=batch.get("op_position"),
        )

        aux_weight = self.config.aux_loss_weight
        if phase == 3:
            aux_weight *= self.config.aux_loss_decay

        if phase == 1:
            # Extraction-only: use only auxiliary loss
            total_loss = aux_weight * aux_loss
        else:
            # Phase 2 & 3: LM loss + weighted aux loss
            total_loss = lm_loss + aux_weight * aux_loss

        return total_loss, lm_loss.item(), aux_loss.item()

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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
        sampler_override=None,
    ) -> DataLoader:
        """Clone a DataLoader with deterministic sampling settings."""
        kwargs = {
            "dataset": loader.dataset,
            "batch_size": loader.batch_size,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "multiprocessing_context": loader.multiprocessing_context,
            "persistent_workers": loader.persistent_workers,
            "pin_memory_device": loader.pin_memory_device,
        }
        if sampler_override is not None:
            kwargs["sampler"] = sampler_override
        else:
            kwargs["shuffle"] = shuffle
            kwargs["generator"] = generator
        if loader.num_workers > 0:
            kwargs["prefetch_factor"] = loader.prefetch_factor
        return DataLoader(**kwargs)

    def _get_curriculum_max_digits(self, epoch: int) -> int | None:
        """Return the curriculum digit cap for the current epoch, or None."""
        schedule = self.config.curriculum_schedule
        if not schedule or self._current_phase != 1:
            return None
        phase1_progress = epoch / max(self.config.phase1_epochs, 1)
        max_digits = schedule[0][1]
        for frac, digits in schedule:
            if phase1_progress >= frac:
                max_digits = digits
        return max_digits

    def _get_filtered_indices(self, epoch: int) -> list[int] | None:
        """Compute eligible sample indices for Phase 1 (aux + curriculum)."""
        if self._current_phase != 1:
            return None

        indices = None

        # Aux-only filtering
        if self.config.phase1_aux_only and self._aux_indices:
            indices = set(self._aux_indices)

        # Curriculum filtering
        max_digits = self._get_curriculum_max_digits(epoch)
        if max_digits is not None and self._max_operand_digits is not None:
            eligible = torch.where(
                (self._max_operand_digits <= max_digits)
                & (self._max_operand_digits > 0)  # exclude non-operand examples
            )[0].tolist()
            if indices is not None:
                indices = indices & set(eligible)
            else:
                indices = set(eligible)

        return sorted(indices) if indices is not None else None

    def _build_epoch_train_loader(self, epoch: int) -> DataLoader:
        """Build a deterministic train loader for the requested epoch."""
        filtered_indices = self._get_filtered_indices(epoch)
        if filtered_indices is not None and len(filtered_indices) > 0:
            # Phase 1 with filtering: use SubsetRandomSampler
            generator = torch.Generator()
            generator.manual_seed(_TRAIN_SAMPLER_SEED + epoch)
            # Shuffle the indices deterministically
            perm = torch.randperm(len(filtered_indices), generator=generator)
            shuffled = [filtered_indices[i] for i in perm]
            sampler = SubsetRandomSampler(shuffled)
            # Update steps_per_epoch for the filtered dataset size
            batch_size = self.train_loader.batch_size or 1
            self.steps_per_epoch = max(
                1, len(filtered_indices) // batch_size // self.accumulation_steps
            )
            return self._clone_loader(
                self.train_loader, shuffle=False, sampler_override=sampler
            )

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

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

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
            "aux_loss": [],
            "lm_loss": [],
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

            # Configure phase at start of each epoch
            phase = self._get_phase(epoch)
            self._configure_phase(phase)

            epoch_loss, epoch_completed = self._train_epoch(epoch, history, step_limit)
            if epoch_loss is not None:
                made_progress = True
                epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}", phase=phase)

            if not epoch_completed:
                break

            epoch_loss_value = epoch_loss if epoch_loss is not None else 0.0
            if self.eval_loader is not None and phase > 1:
                # LM evaluation is only meaningful when injection is active (Phase 2+)
                eval_loss = self._evaluate()
                history["eval_loss"].append(eval_loss)
                epoch_bar.set_postfix(loss=f"{epoch_loss_value:.4f}", eval=f"{eval_loss:.4f}", phase=phase)

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
            elif self.eval_loader is not None and phase == 1:
                # Phase 1: evaluate extraction quality only (no early stopping)
                # Skip eval during early curriculum — results are misleading when
                # model only trains on small digits but eval uses the full range.
                curriculum_max = self._get_curriculum_max_digits(epoch)
                full_digits = self.config.curriculum_schedule[-1][1] if self.config.curriculum_schedule else None
                if curriculum_max is None or curriculum_max >= (full_digits or 0):
                    aux_eval = self._evaluate_extraction()
                    epoch_bar.set_postfix(loss=f"{epoch_loss_value:.4f}", aux_eval=f"{aux_eval:.4f}", phase=phase)
                else:
                    epoch_bar.set_postfix(loss=f"{epoch_loss_value:.4f}", phase=phase)

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
        total_lm_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0
        phase = self._current_phase

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
            desc=f"Epoch {epoch + 1} [P{phase}]",
            unit="batch",
            position=1,
            leave=False,
        )

        while (
            self.batches_completed_in_epoch < self.steps_per_epoch
            and self.global_step < step_limit
            and self.global_step < self.total_training_steps
        ):
            # --- Gradient accumulation loop ---
            step_loss = 0.0
            step_lm = 0.0
            step_aux = 0.0
            micro_counted = 0
            for _micro in range(self.accumulation_steps):
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
                    loss, lm_val, aux_val = self._compute_loss(outputs, batch, phase)

                scaled_loss = loss / self.accumulation_steps
                if scaled_loss.requires_grad:
                    scaled_loss.backward()

                step_loss += loss.item()
                step_lm += lm_val
                step_aux += aux_val
                micro_counted += 1

            if micro_counted == 0:
                break

            # Average over actual micro-batches (handles incomplete final accumulation)
            step_loss /= micro_counted
            step_lm /= micro_counted
            step_aux /= micro_counted

            lr = self._set_learning_rate(self.global_step + 1)
            torch.nn.utils.clip_grad_norm_(self.arb_params, self.config.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += step_loss
            total_lm_loss += step_lm
            total_aux_loss += step_aux
            num_batches += 1
            self.global_step += 1
            self.batches_completed_in_epoch += 1

            avg = total_loss / num_batches
            avg_aux = total_aux_loss / num_batches
            batch_bar.update(1)
            batch_bar.set_postfix(
                loss=f"{avg:.4f}",
                aux=f"{avg_aux:.4f}",
                lr=f"{lr:.2e}",
                step=self.global_step,
            )

            if self.global_step % self.config.log_every == 0:
                history["train_loss"].append(avg)
                history["learning_rate"].append(lr)
                history["aux_loss"].append(avg_aux)
                history["lm_loss"].append(total_lm_loss / num_batches)

            if (
                self.config.checkpoint_every_steps > 0
                and self.global_step % self.config.checkpoint_every_steps == 0
            ):
                self._save_checkpoint(f"step_{self.global_step}")

            if (
                self.eval_loader is not None
                and self.config.eval_every > 0
                and self.global_step % self.config.eval_every == 0
                and phase > 1  # LM eval meaningless in Phase 1
            ):
                eval_loss = self._evaluate()
                history["eval_loss"].append(eval_loss)
                batch_bar.set_postfix(
                    loss=f"{avg:.4f}",
                    aux=f"{avg_aux:.4f}",
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
        """Evaluate on the eval set (capped by max_eval_batches if set)."""
        self.model.eval()
        total_loss = torch.zeros(1, device=self.device)
        num_batches = 0
        max_batches = self.config.max_eval_batches or len(self.eval_loader)

        eval_bar = tqdm(
            self.eval_loader,
            desc="Evaluating",
            unit="batch",
            total=min(max_batches, len(self.eval_loader)),
            position=2,
            leave=False,
        )

        with self._autocast_context():
            for batch in eval_bar:
                if num_batches >= max_batches:
                    break
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss += outputs["loss"].detach()
                num_batches += 1

        eval_bar.close()
        return (total_loss / max(num_batches, 1)).item()

    @torch.no_grad()
    def _evaluate_extraction(self) -> float:
        """Evaluate auxiliary extraction loss on the eval set (Phase 1 metric)."""
        self.model.eval()
        total_aux = 0.0
        num_batches = 0
        max_batches = self.config.max_eval_batches or len(self.eval_loader)

        with self._autocast_context():
            for batch in self.eval_loader:
                if num_batches >= max_batches:
                    break
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                aux_loss = compute_extraction_loss(
                    arb_extractions=outputs.get("arb_extractions", {}),
                    gt_digits_a=batch["digits_a"],
                    gt_digits_b=batch["digits_b"],
                    has_aux=batch["has_aux"],
                    attention_mask=batch["attention_mask"],
                    eq_positions=batch["eq_position"],
                    op_positions=batch.get("op_position"),
                )
                total_aux += aux_loss.item()
                num_batches += 1

        self.model.train()
        return total_aux / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

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
            "current_phase": self._current_phase,
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

        self.global_step = ckpt["global_step"]
        saved_steps_per_epoch = ckpt.get("steps_per_epoch", self.steps_per_epoch)

        if "completed_epochs" in ckpt:
            self.completed_epochs = ckpt["completed_epochs"]
            self.batches_completed_in_epoch = ckpt.get("batches_completed_in_epoch", 0)
        else:
            self.completed_epochs = self.global_step // max(saved_steps_per_epoch, 1)
            self.batches_completed_in_epoch = self.global_step % max(saved_steps_per_epoch, 1)

        self._restore_rng_state(ckpt.get("rng_state"))

        # Restore phase and rebuild optimizer before loading optimizer state
        saved_phase = ckpt.get("current_phase", 0)
        if saved_phase > 0:
            self._current_phase = 0  # force reconfigure
            self._configure_phase(saved_phase)

        # Restore early stopping state AFTER phase config (which resets them)
        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        self.patience_counter = ckpt.get("patience_counter", 0)

        # Load optimizer state after rebuilding with correct param groups
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError):
            logger.warning(
                "Could not restore optimizer state (likely phase transition "
                "changed param groups). Starting with fresh optimizer."
            )

        if saved_steps_per_epoch != self.steps_per_epoch:
            logger.warning(
                "Checkpoint was created with %s steps/epoch, current run has %s. "
                "Mid-epoch resume may be inaccurate if the dataset changed.",
                saved_steps_per_epoch,
                self.steps_per_epoch,
            )

        logger.info(
            "Loaded checkpoint from %s (step %s, epoch %s, batch %s, phase %s)",
            path,
            self.global_step,
            self.completed_epochs,
            self.batches_completed_in_epoch,
            self._current_phase,
        )
