"""AdamW with decay/no-decay parameter groups, and the warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from minimegatron.training.config import TrainConfig


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into a decayed and an undecayed group.

    Weight decay belongs on the matmul weights and embeddings (>= 2-D). Applying it to biases and
    to norm gamma/beta (1-D) pulls them toward zero, which just fights the normalization they
    exist to perform, and buys no regularization.

    `named_parameters()` de-duplicates shared tensors, so a tied embedding/LM-head weight lands in
    exactly one group and is decayed once.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(decay) + len(no_decay) == trainable, "every parameter must be in exactly one group"
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup to `max_lr`, then cosine decay to `min_lr`. A pure function, so it is
    directly unit-testable at the boundaries.

    Warmup exists because Adam's second-moment estimate `v` is built from very few samples in the
    first steps, so early updates are both large and unreliable; ramping the step size keeps them
    from damaging the initialization. Cosine decay then anneals smoothly rather than in jumps.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if max_steps <= warmup_steps:  # degenerate schedule: warmup covers the whole run
        return max_lr
    progress = min((step - warmup_steps) / (max_steps - warmup_steps), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW: decay applied to the weights directly, not added to the gradient.

    Adam divides the update by sqrt(v), so an L2 penalty folded into the gradient gets rescaled
    per-parameter by the optimizer and stops behaving like weight decay at all. AdamW keeps the
    two separate, which is the whole point of the 'W'.

    Memory note for Milestone 3: m and v are a full float32 copy of the model each, so float32
    params + grads + Adam state is 16 bytes per parameter before any activations.
    """
    return torch.optim.AdamW(
        param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )


def set_lr_for_step(optimizer: torch.optim.Optimizer, cfg: TrainConfig, step: int) -> None:
    """Force the optimizer onto the learning rate this config's schedule prescribes at `step`.

    Needed on resume, and only because of an ordering subtlety worth knowing:

    * `LRScheduler.load_state_dict` restores `last_epoch` but does *not* write a learning rate
      into the optimizer -- that only happens inside `step()`.
    * The learning rate nevertheless survives a resume, because the *optimizer's* `state_dict`
      carries its param groups, `lr` included.

    So the rate comes back -- but it is the rate the **previous** run's schedule produced. Resume
    with a changed schedule (`--set train.max_steps=...`, a longer warmup) and the first step
    after resuming would silently use the old curve's value before snapping onto the new one.
    Computing it from the live config instead makes the step after a resume correct in both cases.
    """
    lr = lr_at(step, cfg.lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> LambdaLR:
    """Wrap `lr_at` in a LambdaLR, which gives us a standard state_dict for checkpointing."""

    def factor(step: int) -> float:
        return lr_at(step, cfg.lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps) / cfg.lr

    return LambdaLR(optimizer, lr_lambda=factor)
