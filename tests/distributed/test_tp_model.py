"""The whole model under tensor parallelism, against the single-device reference."""

import pytest
import torch

from minimegatron.distributed.config import DistributedConfig
from minimegatron.distributed.setup import init_distributed
from minimegatron.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    tp_shard_dim,
)
from minimegatron.model.config import ModelConfig
from minimegatron.model.transformer import GPT
from minimegatron.training.checkpoint import load_checkpoint
from minimegatron.training.trainer import Trainer
from tests.distributed.conftest import (
    NANO_MODEL,
    nano_experiment,
    run_distributed,
    single_process_reference,
)

pytestmark = pytest.mark.distributed

MICRO = 8  # every TP rank sees the SAME batch, so this matches the reference's micro batch


def _tp_run(rank, world_size, out_dir, steps):
    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, strategy="tp", micro_batch_size=MICRO, warmup_steps=2)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    records = [trainer.train_step() for _ in range(steps)]

    block = trainer.raw_model.blocks[0]
    return {
        "records": records,
        "grads": {n: p.grad.clone() for n, p in trainer.raw_model.named_parameters()},
        "weights": {n: p.detach().clone() for n, p in trainer.raw_model.named_parameters()},
        "local_heads": block.attn.n_heads,
        "layer_types": {
            "qkv": type(block.attn.qkv).__name__,
            "out_proj": type(block.attn.out_proj).__name__,
            "fc1": type(block.mlp.fc1).__name__,
            "fc2": type(block.mlp.fc2).__name__,
        },
        "dp_world_size": trainer.dp_world_size,
        "tp_size": trainer.tp_size,
        "tokens_seen": trainer.tokens_seen,
    }


def _reference(tmp_path, steps):
    return single_process_reference(
        str(tmp_path / "ref"), steps=steps, micro_batch_size=MICRO, warmup_steps=2
    )


# --- structure ----------------------------------------------------------------------------------


def test_the_right_layers_were_replaced(tmp_path):
    """fc1/qkv column parallel, fc2/out_proj row parallel -- one column->row pair each."""
    results = run_distributed(_tp_run, 2, args=(str(tmp_path / "tp"), 1))
    assert results[0]["layer_types"] == {
        "qkv": ColumnParallelLinear.__name__,
        "out_proj": RowParallelLinear.__name__,
        "fc1": ColumnParallelLinear.__name__,
        "fc2": RowParallelLinear.__name__,
    }


def test_each_rank_runs_half_the_attention_heads(tmp_path):
    """Why the per-head QKV layout matters (D6): each rank owns complete heads, so attention
    itself needs no communication. `n_heads` must be the *local* count or `split_qkv` would
    reshape this rank's slice with the wrong head count."""
    results = run_distributed(_tp_run, 2, args=(str(tmp_path / "tp"), 1))
    for rank in (0, 1):
        assert results[rank]["local_heads"] == NANO_MODEL["n_heads"] // 2


def test_tp_ranks_are_not_data_parallel(tmp_path):
    """TP ranks jointly compute one forward pass over one batch, so the data-parallel world is 1
    and tokens/s must not be multiplied by tp_size."""
    results = run_distributed(_tp_run, 2, args=(str(tmp_path / "tp"), 1))
    for rank in (0, 1):
        assert results[rank]["tp_size"] == 2
        assert results[rank]["dp_world_size"] == 1
        assert results[rank]["tokens_seen"] == MICRO * NANO_MODEL["seq_len"]


# --- forward and loss ---------------------------------------------------------------------------


def test_tp_loss_matches_the_single_device_reference(tmp_path):
    """Exit criterion 3."""
    results = run_distributed(_tp_run, 2, args=(str(tmp_path / "tp"), 1))
    reference, _, _ = _reference(tmp_path, 1)
    for rank in (0, 1):
        assert results[rank]["records"][0]["loss"] == pytest.approx(
            reference[0]["loss"], rel=1e-5
        )


def test_gradient_norm_matches_the_reference(tmp_path):
    results = run_distributed(_tp_run, 2, args=(str(tmp_path / "tp"), 1))
    reference, _, _ = _reference(tmp_path, 1)
    assert results[0]["records"][0]["grad_norm"] == pytest.approx(
        reference[0]["grad_norm"], rel=1e-4
    )


# --- gradients: sharded slices and replicated wholes --------------------------------------------


def test_sharded_parameter_gradients_match_the_reference_slice(tmp_path):
    """Exit criterion 4, first half. Driven by `tp_shard_dim`, the same function that shards a
    state dict, so the test and the implementation agree on what is split where."""
    results = run_distributed(_tp_run, 2, args=(str(tmp_path / "tp"), 1))
    _, reference_grads, _ = _reference(tmp_path, 1)

    checked = 0
    for rank in (0, 1):
        for name, grad in results[rank]["grads"].items():
            dim = tp_shard_dim(name)
            if dim is None:
                continue
            expected = reference_grads[name].chunk(2, dim=dim)[rank]
            torch.testing.assert_close(
                grad, expected, rtol=1e-4, atol=1e-6, msg=f"{name} on rank {rank}"
            )
            checked += 1
    assert checked > 0, "no sharded parameters were compared"


def test_replicated_parameter_gradients_match_in_full_and_agree_across_ranks(tmp_path):
    """Exit criterion 4, second half -- and a sharper test than it looks.

    A replicated parameter's gradient is only complete if every rank's contribution was summed,
    which happens in `f`'s backward. A missing all-reduce there would leave each rank with a
    partial and the ranks disagreeing.
    """
    results = run_distributed(_tp_run, 2, args=(str(tmp_path / "tp"), 1))
    _, reference_grads, _ = _reference(tmp_path, 1)

    checked = 0
    for name, grad in results[0]["grads"].items():
        if tp_shard_dim(name) is not None:
            continue
        torch.testing.assert_close(
            grad, reference_grads[name], rtol=1e-4, atol=1e-6, msg=f"{name} vs reference"
        )
        torch.testing.assert_close(grad, results[1]["grads"][name], msg=f"{name} across ranks")
        checked += 1
    assert checked > 0, "no replicated parameters were compared"


def test_the_tied_embedding_gradient_is_complete_on_every_rank(tmp_path):
    """The tied table is replicated and used twice per forward; both uses must reach both ranks."""
    results = run_distributed(_tp_run, 2, args=(str(tmp_path / "tp"), 1))
    _, reference_grads, _ = _reference(tmp_path, 1)
    for rank in (0, 1):
        torch.testing.assert_close(
            results[rank]["grads"]["tok_emb.weight"],
            reference_grads["tok_emb.weight"],
            rtol=1e-4,
            atol=1e-6,
        )


# --- training trajectory ------------------------------------------------------------------------


def test_tp_tracks_the_reference_over_several_steps(tmp_path):
    """Exit criterion 5. Several steps, so an error that needs the weights to drift apart before
    it shows up cannot hide behind a single comparison."""
    steps = 5
    results = run_distributed(_tp_run, 2, args=(str(tmp_path / "tp"), steps))
    reference, _, _ = _reference(tmp_path, steps)

    got = [r["loss"] for r in results[0]["records"]]
    want = [r["loss"] for r in reference]
    for step, (a, b) in enumerate(zip(got, want)):
        assert a == pytest.approx(b, rel=1e-4), f"diverged at step {step}"


def test_weights_match_the_reference_after_several_steps(tmp_path):
    steps = 5
    results = run_distributed(_tp_run, 2, args=(str(tmp_path / "tp"), steps))
    _, _, ref_trainer = _reference(tmp_path, steps)
    reference = dict(ref_trainer.raw_model.named_parameters())

    for name, weight in results[0]["weights"].items():
        dim = tp_shard_dim(name)
        expected = reference[name].detach()
        if dim is not None:
            expected = expected.chunk(2, dim=dim)[0]
        # Looser than the loss comparison on purpose, and the loosening is an *absolute* floor
        # rather than a relative one. The parallel matmuls sum in a different order, and Adam
        # divides by sqrt(v), so a ~1e-8 gradient difference becomes ~1e-6 in a parameter. The
        # affected parameters are the biases, which start at exactly zero, so a relative
        # tolerance buys nothing for the elements still near zero. Measured worst case over 5
        # steps: 1.4e-6 absolute, against a largest bias of 4.1e-3 -- so 1e-5 is still 0.25% of
        # the parameter scale, not a blanket "loosen until green".
        torch.testing.assert_close(
            weight,
            expected,
            rtol=1e-3,
            atol=1e-5,
            msg=lambda default, name=name: f"{name}: {default}",
        )


# --- checkpoints --------------------------------------------------------------------------------


def _tp_train_and_save(rank, world_size, out_dir):
    from pathlib import Path

    info = init_distributed(DistributedConfig(backend="gloo"))
    cfg = nano_experiment(out_dir, strategy="tp", micro_batch_size=MICRO, max_steps=3)
    trainer = Trainer(cfg, dist_info=info, out_dir=out_dir, echo=False)
    trainer.fit()
    return {
        "files": sorted(p.name for p in Path(out_dir).iterdir()),
        "step": trainer.step,
    }


def test_tp_writes_a_per_rank_and_a_gathered_checkpoint(tmp_path):
    out_dir = tmp_path / "run"
    results = run_distributed(_tp_train_and_save, 2, args=(str(out_dir),))
    # Per-rank files resume the run (each rank owns different slices and its own Adam moments);
    # the gathered file makes the result loadable anywhere.
    assert results[0]["files"] == [
        "ckpt_000003_full.pt",
        "ckpt_000003_tp0.pt",
        "ckpt_000003_tp1.pt",
        "log.jsonl",
    ]


def test_the_gathered_tp_checkpoint_loads_into_a_single_device_model(tmp_path):
    """Exit criterion 6, end to end: a TP checkpoint gathers back to a full model."""
    out_dir = tmp_path / "run"
    run_distributed(_tp_train_and_save, 2, args=(str(out_dir),))

    plain = GPT(ModelConfig(dropout=0.0, **NANO_MODEL))
    meta = load_checkpoint(out_dir / "ckpt_000003_full.pt", model=plain)

    assert meta["step"] == 3
    assert meta["tokens_seen"] == 3 * MICRO * NANO_MODEL["seq_len"]  # dp world is 1
    assert plain.lm_head.weight is plain.tok_emb.weight  # tying intact


def _tp_resume(rank, world_size, out_dir):
    info = init_distributed(DistributedConfig(backend="gloo"))
    steps = 3

    def build(sub):
        cfg = nano_experiment(
            f"{out_dir}/{sub}", strategy="tp", micro_batch_size=MICRO, max_steps=2 * steps
        )
        return Trainer(cfg, dist_info=info, out_dir=f"{out_dir}/{sub}", echo=False)

    straight = build("a")
    straight_losses = [straight.train_step()["loss"] for _ in range(2 * steps)]

    first = build("b")
    resumed = [first.train_step()["loss"] for _ in range(steps)]
    ckpt = first.save()

    second = build("c")
    meta = second.resume(ckpt)
    resumed += [second.train_step()["loss"] for _ in range(steps)]

    return {"straight": straight_losses, "resumed": resumed, "step": meta["step"]}


def test_resuming_a_tp_run_from_its_per_rank_checkpoint(tmp_path):
    results = run_distributed(_tp_resume, 2, args=(str(tmp_path),))
    for rank in (0, 1):
        assert results[rank]["step"] == 3
        assert results[rank]["resumed"] == pytest.approx(results[rank]["straight"], rel=1e-6)


def test_the_gathered_checkpoint_can_actually_resume_a_single_device_run(tmp_path):
    """Regression for ISSUES #17: the consolidated file must be *internally consistent*.

    Gathering the model but not the optimizer produced full weights beside half-width Adam
    moments. That file loaded without complaint and then died on the first optimizer step with
    `The size of tensor a (288) must match the size of tensor b (576)` -- a shape error naming
    nothing actionable, one step into a resumed run.

    So this does not stop at loading: it takes an optimizer step, which is where the inconsistency
    actually surfaced.
    """
    import torch as _torch

    from minimegatron.training.config import TrainConfig
    from minimegatron.training.optimizer import build_optimizer

    out_dir = tmp_path / "run"
    run_distributed(_tp_train_and_save, 2, args=(str(out_dir),))

    plain = GPT(ModelConfig(dropout=0.0, **NANO_MODEL))
    optimizer = build_optimizer(plain, TrainConfig(lr=1e-3))
    meta = load_checkpoint(
        out_dir / "ckpt_000003_full.pt", model=plain, optimizer=optimizer
    )
    assert meta["step"] == 3

    # Every restored moment must match its parameter's full shape, not a shard of it.
    flat = [p for group in optimizer.param_groups for p in group["params"]]
    for index, param in enumerate(flat):
        entry = optimizer.state_dict()["state"].get(index)
        if entry is None:
            continue
        assert entry["exp_avg"].shape == param.shape, f"param {index} has a sharded moment"

    idx = _torch.randint(0, NANO_MODEL["vocab_size"], (2, 8))
    _, loss = plain(idx, idx)
    loss.backward()
    optimizer.step()  # this is the line that used to raise
    assert _torch.isfinite(loss)
