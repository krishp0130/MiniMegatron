"""The single-device training loop.

This is the reference every later milestone is measured against: DDP (M2), FSDP2 (M3) and tensor
parallelism (M4) must all reproduce the losses and gradients this loop produces.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from minimegatron.config import ConfigError
from minimegatron.data.synthetic import BatchStream, build_dataset
from minimegatron.model.transformer import GPT
from minimegatron.training.checkpoint import load_checkpoint, save_checkpoint
from minimegatron.training.metrics import JSONLLogger
from minimegatron.training.optimizer import build_optimizer, build_scheduler, set_lr_for_step

if TYPE_CHECKING:  # `experiment` imports training.config, so importing it here at runtime would
    from minimegatron.experiment import ExperimentConfig  # close an import cycle.


def resolve_device(spec: str) -> torch.device:
    """Resolve a device string, with no silent fallback.

    `auto` prefers cuda, then mps, then cpu. An explicit `cuda` on a machine without CUDA is a
    hard error: a run that quietly drops to CPU looks like it worked and reports throughput that
    is wrong by two orders of magnitude.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError("device 'cuda' requested but CUDA is not available")
    if spec == "mps" and not torch.backends.mps.is_available():
        raise ConfigError("device 'mps' requested but MPS is not available")
    return torch.device(spec)


def synchronize(device: torch.device) -> None:
    """Wait for queued kernels to finish before reading a clock.

    CUDA and MPS calls are asynchronous: Python enqueues work and returns immediately. Timing a
    step without synchronizing measures how fast we can enqueue, not how fast the device computes.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators


class Trainer:
    """Owns the model, optimizer, schedule, data streams, logging, and checkpoints."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        dp_world_size: int = 1,
        out_dir: str | Path | None = None,
        echo: bool = True,
    ) -> None:
        cfg.train.validate_global_batch(dp_world_size)  # refuse to start, don't warn
        self.cfg = cfg
        self.out_dir = Path(out_dir if out_dir is not None else cfg.train.out_dir)

        seed_everything(cfg.train.seed)
        self.device = resolve_device(cfg.train.device)

        self.model = GPT(cfg.model).to(self.device)
        self.optimizer = build_optimizer(self.model, cfg.train)
        self.scheduler = build_scheduler(self.optimizer, cfg.train)

        micro = cfg.train.micro_batch_size
        self.train_stream = BatchStream(
            build_dataset(cfg.data, cfg.model, "train"), micro, cfg.data.seed
        )
        self.val_stream = BatchStream(
            build_dataset(cfg.data, cfg.model, "val"), micro, cfg.data.val_seed
        )

        self.step = 0
        self.tokens_seen = 0
        self.logger = JSONLLogger(self.out_dir / "log.jsonl", echo=echo)
        self._wall_start = time.perf_counter()
        self._mark_time = self._wall_start
        self._mark_tokens = 0

    # --- one optimizer step ---------------------------------------------------------------------

    def train_step(self) -> dict[str, Any]:
        """Run `grad_accum_steps` micro-batches, then one optimizer step."""
        cfg = self.cfg.train
        accum = cfg.grad_accum_steps
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss_sum = torch.zeros((), device=self.device)
        for j in range(accum):
            # Micro-batch index is global: the samples consumed by step s depend only on the seed
            # and on how many samples came before, not on how they were grouped.
            x, y = self.train_stream.micro_batch(self.step * accum + j)
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            # Gradients SUM across backward() calls, so scaling each micro-loss by 1/accum makes
            # the accumulated gradient exactly the gradient of the mean loss over the global
            # batch -- identical to one large batch, up to float rounding.
            (loss / accum).backward()
            loss_sum += loss.detach()  # kept on device: one sync per step, not one per micro-batch

        # The pre-clip norm is the headline instability signal, so it is logged every step even
        # when clipping is disabled (max_norm=inf measures without rescaling anything).
        max_norm = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
        grad_norm = clip_grad_norm_(self.model.parameters(), max_norm)

        lr = self.optimizer.param_groups[0]["lr"]  # the LR this step actually used
        self.optimizer.step()
        self.scheduler.step()

        self.step += 1
        self.tokens_seen += accum * cfg.micro_batch_size * self.cfg.model.seq_len
        return {
            "step": self.step,
            "loss": (loss_sum / accum).item(),
            "lr": lr,
            "grad_norm": float(grad_norm),
            "tokens_seen": self.tokens_seen,
        }

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean loss over `eval_steps` fixed validation batches.

        The batches are the same at every evaluation, so successive validation losses are directly
        comparable. `eval()` disables dropout, which also means evaluation consumes no RNG and so
        cannot perturb a resumed run.
        """
        self.model.eval()
        total = torch.zeros((), device=self.device)
        for j in range(self.cfg.train.eval_steps):
            x, y = self.val_stream.micro_batch(j)
            _, loss = self.model(x.to(self.device), y.to(self.device))
            total += loss
        self.model.train()
        return (total / self.cfg.train.eval_steps).item()

    # --- the run --------------------------------------------------------------------------------

    def fit(self) -> dict[str, Any]:
        cfg = self.cfg.train
        record: dict[str, Any] = {}
        while self.step < cfg.max_steps:
            record = self.train_step()
            is_last = self.step >= cfg.max_steps

            if cfg.log_interval and (self.step % cfg.log_interval == 0 or is_last):
                record.update(self._throughput())
                if cfg.eval_interval and (self.step % cfg.eval_interval == 0 or is_last):
                    record["val_loss"] = self.evaluate()
                self.logger.log(record)
            elif cfg.eval_interval and self.step % cfg.eval_interval == 0:
                self.logger.log({**record, **self._throughput(), "val_loss": self.evaluate()})

            if cfg.ckpt_interval and self.step % cfg.ckpt_interval == 0 and not is_last:
                self.save()

        self.save()
        self.logger.close()
        return record

    def _throughput(self) -> dict[str, float]:
        """Tokens/s since the last log boundary, measured after a device synchronization."""
        synchronize(self.device)
        now = time.perf_counter()
        window = now - self._mark_time
        tokens = self.tokens_seen - self._mark_tokens
        self._mark_time, self._mark_tokens = now, self.tokens_seen
        return {
            "tokens_per_sec": tokens / window if window > 0 else float("nan"),
            "elapsed": now - self._wall_start,
        }

    # --- checkpoints ----------------------------------------------------------------------------

    def checkpoint_path(self, step: int | None = None) -> Path:
        return self.out_dir / f"ckpt_{self.step if step is None else step:06d}.pt"

    def save(self, path: str | Path | None = None) -> Path:
        return save_checkpoint(
            self.checkpoint_path() if path is None else path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            tokens_seen=self.tokens_seen,
            config=self.cfg,
            device=self.device,
        )

    def resume(self, path: str | Path) -> dict[str, Any]:
        """Restore a run exactly. The data position needs no state: the sample stream is a pure
        function of (seed, samples consumed), and `step` tells us how many that is."""
        meta = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        self.step = meta["step"]
        self.tokens_seen = meta["tokens_seen"]
        # The restored rate is the one the *previous* run's schedule produced; recompute it from
        # this run's config so resuming with a changed schedule is correct from the first step.
        set_lr_for_step(self.optimizer, self.cfg.train, self.step)
        self._mark_tokens = self.tokens_seen
        self._mark_time = time.perf_counter()
        return meta
