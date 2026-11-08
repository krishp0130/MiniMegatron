"""The single-device training loop.

This is the reference every later milestone is measured against: DDP (M2), FSDP2 (M3) and tensor
parallelism (M4) must all reproduce the losses and gradients this loop produces.
"""

from __future__ import annotations

import contextlib
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from minimegatron.config import ConfigError
from minimegatron.data.synthetic import BatchStream, build_dataset
from minimegatron.distributed.checkpoint import load_sharded, save_consolidated, save_sharded
from minimegatron.distributed.fsdp import build_fsdp, precision_dtype
from minimegatron.distributed.setup import (
    DistInfo,
    all_reduce_mean,
    barrier,
    build_device_mesh,
    resolve_strategy,
)
from minimegatron.model.transformer import GPT
from minimegatron.training.checkpoint import load_checkpoint, save_checkpoint
from minimegatron.training.metrics import JSONLLogger, NullLogger
from minimegatron.training.optimizer import build_optimizer, build_scheduler, set_lr_for_step

if TYPE_CHECKING:  # `experiment` imports training.config, so importing it here at runtime would
    from minimegatron.experiment import ExperimentConfig  # close an import cycle.


def resolve_device(spec: str, dist_info: DistInfo | None = None) -> torch.device:
    """Resolve a device string, with no silent fallback.

    `auto` prefers cuda, then mps, then cpu. An explicit `cuda` on a machine without CUDA is a
    hard error: a run that quietly drops to CPU looks like it worked and reports throughput that
    is wrong by two orders of magnitude.

    Under distributed CUDA the device is `cuda:{local_rank}`, so each process owns exactly one
    GPU. Without the index every rank would pile onto cuda:0.
    """
    local_rank = dist_info.local_rank if dist_info is not None else 0
    distributed = dist_info is not None and dist_info.is_distributed

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}" if distributed else "cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError("device 'cuda' requested but CUDA is not available")
    if spec == "mps" and not torch.backends.mps.is_available():
        raise ConfigError("device 'mps' requested but MPS is not available")
    if spec == "cuda" and distributed:
        return torch.device(f"cuda:{local_rank}")
    return torch.device(spec)


def validate_precision(precision: str, device: torch.device) -> None:
    """Refuse bf16 on hardware that cannot do it, rather than quietly changing the numerics.

    bf16 needs Ampere (sm_80) or newer. This matters concretely: the T4s on the free tier of most
    notebook services are sm_75 and have no bf16 at all, so a silent fallback would produce a run
    whose precision does not match its config -- and whose FP32-vs-BF16 comparison would be
    meaningless. CPU and MPS both support bf16 here, so only CUDA needs the check.
    """
    if precision != "bf16":
        return
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        name = torch.cuda.get_device_name(device)
        raise ConfigError(
            f"precision 'bf16' requested but {name} does not support bfloat16 "
            "(needs compute capability 8.0+); use precision 'fp32'"
        )


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
        dist_info: DistInfo | None = None,
        out_dir: str | Path | None = None,
        echo: bool = True,
    ) -> None:
        self.dist = dist_info if dist_info is not None else DistInfo()
        cfg.train.validate_global_batch(self.dist.world_size)  # refuse to start, don't warn
        self.strategy = resolve_strategy(cfg.distributed.strategy, self.dist.world_size)
        self.cfg = cfg
        self.out_dir = Path(out_dir if out_dir is not None else cfg.train.out_dir)

        # Every rank seeds identically, so the replicas start from the same weights. DDP also
        # broadcasts rank 0's parameters at construction, so this is belt and braces -- but it
        # keeps a single-process run and rank 0 of a distributed run bit-identical at init.
        seed_everything(cfg.train.seed)
        self.device = resolve_device(cfg.train.device, self.dist)
        validate_precision(cfg.train.precision, self.device)
        # None for fp32; the autocast compute dtype for bf16. Parameters stay fp32 either way.
        self.autocast_dtype = precision_dtype(cfg.train.precision)

        # `raw_model` is always the real GPT; `model` may be a wrapper around it. Keeping both
        # means parameter access, clipping and checkpointing never depend on the wrapper. (Under
        # FSDP they are the same object: `fully_shard` modifies the module in place.)
        self.raw_model = GPT(cfg.model).to(self.device)
        self.model: nn.Module = self.raw_model
        self.mesh = None

        if self.strategy == "ddp":
            self.model = DistributedDataParallel(
                self.raw_model,
                # One device per process on CUDA; gloo on CPU takes no device_ids.
                device_ids=[self.dist.local_rank] if self.device.type == "cuda" else None,
                # Every parameter receives a gradient every step, including the tied embedding
                # (used twice per forward). Leaving this False keeps the reducer's fast path and
                # turns any future unused parameter into a loud error rather than silent overhead.
                find_unused_parameters=False,
            )
        elif self.strategy == "fsdp":
            # Explicit device type: letting FSDP build a default mesh means auto-detecting the
            # accelerator, which finds MPS here and crashes (ISSUES #2).
            self.mesh = build_device_mesh(self.device.type, self.dist.world_size)
            self.model = build_fsdp(self.raw_model, self.mesh, cfg.train.precision)

        # AFTER wrapping, always: `fully_shard` replaces parameters with sharded DTensors in
        # place, so an optimizer built earlier would hold the pre-shard tensors and silently
        # update nothing that matters.
        self.optimizer = build_optimizer(self.raw_model, cfg.train)
        self.scheduler = build_scheduler(self.optimizer, cfg.train)

        micro = cfg.train.micro_batch_size
        shard = dict(rank=self.dist.rank, world_size=self.dist.world_size)
        self.train_stream = BatchStream(
            build_dataset(cfg.data, cfg.model, "train"), micro, cfg.data.seed, **shard
        )
        self.val_stream = BatchStream(
            build_dataset(cfg.data, cfg.model, "val"), micro, cfg.data.val_seed, **shard
        )

        self.step = 0
        self.tokens_seen = 0
        # Only rank 0 writes anything; see NullLogger for why.
        log_factory = JSONLLogger if self.dist.is_main else NullLogger
        self.logger = log_factory(self.out_dir / "log.jsonl", echo=echo)
        self._wall_start = time.perf_counter()
        self._mark_time = self._wall_start
        self._mark_tokens = 0

    # --- one optimizer step ---------------------------------------------------------------------

    def _autocast(self):
        """Run the forward in bf16 while parameters stay fp32 ("mixed precision").

        The optimizer step, weight decay and clipping stay in fp32 outside this context -- fp32
        master weights are what keep small updates from being rounded away, since bf16 has only 8
        mantissa bits. The loss needs no special handling: `GPT.forward` already promotes logits
        to at least fp32 before cross-entropy (the ISSUES #4 pattern), so the log-sum-exp over the
        vocabulary -- the most precision-sensitive op in the model -- is fp32 for free.

        Not used under FSDP, which has its own mechanism: `MixedPrecisionPolicy` already
        all-gathers parameters as bf16 and casts forward inputs to match, so autocast on top would
        be redundant double-casting. The two are not interchangeable, though -- autocast converts
        at the op boundary, after the gather has already moved fp32 bytes, so only the policy
        actually halves the communication.
        """
        if self.autocast_dtype is None or self.strategy == "fsdp":
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _sync_context(self, is_last_micro_batch: bool):
        """Suppress DDP's gradient all-reduce on all but the last micro-batch of a step.

        Only the gradient after the final micro-batch is used, so all-reducing the intermediate
        partial sums costs `grad_accum_steps` times the communication and discards all but one of
        the results. Correctness is unaffected because local accumulation and all-reduce are both
        sums, and sums commute: allreduce(sum_j g_j) == sum_j allreduce(g_j).
        """
        if is_last_micro_batch:
            return contextlib.nullcontext()
        if isinstance(self.model, DistributedDataParallel):
            return self.model.no_sync()
        if self.strategy == "fsdp":
            return self._fsdp_no_sync()
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _fsdp_no_sync(self):
        """FSDP2's equivalent of DDP's `no_sync`: skip the gradient reduce-scatter.

        `set_is_last_backward(False)` additionally tells FSDP not to run its end-of-backward
        cleanup yet. The trade-off differs from DDP's: unreduced FSDP gradients accumulate
        *unsharded*, so suppressing the collective costs memory to save bandwidth. Correct either
        way, since reduction and accumulation are both sums.
        """
        self.model.set_requires_gradient_sync(False)
        self.model.set_is_last_backward(False)
        try:
            yield
        finally:
            self.model.set_requires_gradient_sync(True)
            self.model.set_is_last_backward(True)

    def train_step(self) -> dict[str, Any]:
        """Run `grad_accum_steps` micro-batches, then one optimizer step."""
        cfg = self.cfg.train
        accum = cfg.grad_accum_steps
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss_sum = torch.zeros((), device=self.device)
        for j in range(accum):
            # Micro-batch index is global: the samples consumed by step s depend only on the seed,
            # the world size and how many samples came before -- not on how they were grouped.
            with self._sync_context(is_last_micro_batch=j == accum - 1):
                x, y = self.train_stream.micro_batch(self.step * accum + j)
                x, y = x.to(self.device), y.to(self.device)
                with self._autocast():
                    _, loss = self.model(x, y)
                # Gradients SUM across backward() calls, so scaling each micro-loss by 1/accum
                # makes the accumulated gradient exactly the gradient of the mean loss over the
                # global batch -- identical to one large batch, up to float rounding.
                (loss / accum).backward()
            loss_sum += loss.detach()  # kept on device: one sync per step, not one per micro-batch

        # Clipping must come *after* the synchronizing backward, and on the unwrapped module's
        # parameters. Clipping per-rank gradients before the all-reduce would give every rank a
        # different norm from its own shard, so the effective threshold would depend on the world
        # size and the data split rather than on the configured value.
        max_norm = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
        grad_norm = clip_grad_norm_(self.raw_model.parameters(), max_norm)

        lr = self.optimizer.param_groups[0]["lr"]  # the LR this step actually used
        self.optimizer.step()
        self.scheduler.step()

        self.step += 1
        self.tokens_seen += (
            accum * cfg.micro_batch_size * self.cfg.model.seq_len * self.dist.world_size
        )
        return {
            "step": self.step,
            # A loss computed here describes this rank's shard; averaging across ranks makes the
            # logged number describe the global batch, which is what it claims to be.
            "loss": all_reduce_mean(loss_sum / accum).item(),
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
            # The unwrapped module: a DDP forward here would only add reducer bookkeeping, and
            # there is no backward to synchronize.
            with self._autocast():
                _, loss = self.raw_model(x.to(self.device), y.to(self.device))
            total += loss
        self.model.train()
        # Each rank evaluated its own shard of equal size, so the mean of the means is the mean.
        return all_reduce_mean(total / self.cfg.train.eval_steps).item()

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
        """Single-file checkpoint path, or a directory under FSDP (DCP writes a directory)."""
        stem = f"ckpt_{self.step if step is None else step:06d}"
        return self.out_dir / (stem if self.strategy == "fsdp" else f"{stem}.pt")

    def consolidated_path(self, step: int | None = None) -> Path:
        """Where the gathered, single-device-compatible file goes alongside the shards."""
        return self.out_dir / f"ckpt_{self.step if step is None else step:06d}_full.pt"

    def save(self, path: str | Path | None = None) -> Path:
        """Write a checkpoint from rank 0 only, then wait for it to exist.

        Every rank holds identical weights after the optimizer step, so `world_size` writes of the
        same bytes is waste. The barrier stops the other ranks racing past a checkpoint that is
        not on disk yet. The *unwrapped* module is saved, so the keys carry no `module.` prefix
        and the file loads straight into a single-device model.
        """
        target = Path(self.checkpoint_path() if path is None else path)

        if self.strategy == "fsdp":
            # Two formats, both written: the sharded directory is what resume uses and is the only
            # one that scales past a single device's memory, while the consolidated file keeps
            # checkpoints interchangeable with single-device and DDP runs -- which is what makes
            # an FSDP run comparable to the reference at all.
            save_sharded(
                target,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=self.step,
                tokens_seen=self.tokens_seen,
                config=self.cfg,
                rank=self.dist.rank,
                is_main=self.dist.is_main,
                device=self.device,
            )
            save_consolidated(
                self.consolidated_path(),
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=self.step,
                tokens_seen=self.tokens_seen,
                config=self.cfg,
                is_main=self.dist.is_main,
                device=self.device,
            )
        elif self.dist.is_main:
            save_checkpoint(
                target,
                model=self.raw_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=self.step,
                tokens_seen=self.tokens_seen,
                config=self.cfg,
                device=self.device,
            )
        barrier()
        return target

    def resume(self, path: str | Path) -> dict[str, Any]:
        """Restore a run exactly. The data position needs no state: the sample stream is a pure
        function of (seed, samples consumed), and `step` tells us how many that is."""
        if Path(path).is_dir():
            meta = load_sharded(
                path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                rank=self.dist.rank,
            )
        else:
            meta = load_checkpoint(
                path,
                model=self.raw_model,
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
