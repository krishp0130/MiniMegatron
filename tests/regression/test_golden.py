"""Deterministic workloads whose behaviour is pinned to recorded values.

These are the **functional** regression tests: hard assertions on facts that must not change
silently. Performance regressions are deliberately *not* here -- they are warnings against a stored
baseline, run by `python -m minimegatron.benchmarks.regression`. SCOPE.md section 8: *"Avoid
treating noisy performance measurements as deterministic unit tests."*

Golden values are legitimate because the two things they depend on are pinned: `torch==2.14.1`
exactly (D1), and index-deterministic data (D9). If a deliberate change to either moves them,
**re-record them deliberately** and say so in the commit message. Do not widen the tolerance to
make a real behavioural change go quiet -- that is the whole point of the test.

Recorded on 2026-11-08, commit 7b8313e, Apple M4 Pro CPU, torch 2.14.1.
"""

import tempfile

import pytest
import torch

from minimegatron.data.config import DataConfig
from minimegatron.experiment import ExperimentConfig, load_config
from minimegatron.model.config import ModelConfig
from minimegatron.model.transformer import GPT
from minimegatron.training.config import TrainConfig
from minimegatron.training.trainer import Trainer
from tests.conftest import REPO_ROOT

NANO = dict(vocab_size=128, seq_len=16, n_layers=2, n_heads=4, d_model=32, d_ff=64, dropout=0.0)
STEPS = 5

GOLDEN_LOSSES = [
    4.858948707580566,
    4.809524059295654,
    4.735809803009033,
    4.615445613861084,
    4.655106544494629,
]
GOLDEN_GRAD_NORMS = [
    1.5139539241790771,
    1.3500580787658691,
    1.180749535560608,
    1.3518283367156982,
    1.2842345237731934,
]
GOLDEN_LRS = [0.0015, 0.003, 0.003, 0.002325, 0.0009750000000000004]

RERECORD = (
    "\n\nIf this change was deliberate (a torch bump, an intentional change to the model, the "
    "schedule or the data), re-record the golden values and say why in the commit message. Do not "
    "loosen the tolerance to silence a real behavioural change."
)

NANO_STATE_DICT_KEYS = [
    "blocks.0.attn.out_proj.bias", "blocks.0.attn.out_proj.weight",
    "blocks.0.attn.qkv.bias", "blocks.0.attn.qkv.weight",
    "blocks.0.mlp.fc1.bias", "blocks.0.mlp.fc1.weight",
    "blocks.0.mlp.fc2.bias", "blocks.0.mlp.fc2.weight",
    "blocks.0.norm1.bias", "blocks.0.norm1.weight",
    "blocks.0.norm2.bias", "blocks.0.norm2.weight",
    "blocks.1.attn.out_proj.bias", "blocks.1.attn.out_proj.weight",
    "blocks.1.attn.qkv.bias", "blocks.1.attn.qkv.weight",
    "blocks.1.mlp.fc1.bias", "blocks.1.mlp.fc1.weight",
    "blocks.1.mlp.fc2.bias", "blocks.1.mlp.fc2.weight",
    "blocks.1.norm1.bias", "blocks.1.norm1.weight",
    "blocks.1.norm2.bias", "blocks.1.norm2.weight",
    "lm_head.weight", "norm_f.bias", "norm_f.weight", "pos_emb.weight", "tok_emb.weight",
]

PRESET_PARAMS = {
    "tiny": (11_487_360, 1_779_840),
    "small": (51_237_888, 25_220_096),
    "medium": (124_475_904, 85_056_000),
}


def golden_experiment(out_dir, **train_overrides) -> ExperimentConfig:
    train = dict(
        max_steps=STEPS, micro_batch_size=4, grad_accum_steps=1, lr=3e-3, warmup_steps=2,
        grad_clip=1.0, device="cpu", precision="fp32", seed=1337, out_dir=str(out_dir),
        log_interval=0, eval_interval=0,
    )
    train.update(train_overrides)
    return ExperimentConfig(
        model=ModelConfig(**NANO),
        data=DataConfig(dataset="memorization", n_items=64, n_sequences=8, seed=0),
        train=TrainConfig(**train),
    )


def run_golden(out_dir, **overrides):
    trainer = Trainer(golden_experiment(out_dir, **overrides), out_dir=out_dir, echo=False)
    return [trainer.train_step() for _ in range(STEPS)], trainer


# --- the pinned workload ------------------------------------------------------------------------


def test_golden_loss_trajectory(tmp_path):
    """The headline invariant: the same config must still train the same way."""
    records, _ = run_golden(tmp_path)
    losses = [r["loss"] for r in records]
    assert losses == pytest.approx(GOLDEN_LOSSES, rel=1e-6), (
        f"loss trajectory changed:\n  recorded {GOLDEN_LOSSES}\n  got      {losses}" + RERECORD
    )


def test_golden_gradient_norms(tmp_path):
    """Pinned separately from the loss: a gradient bug can leave the loss plausible for a few
    steps while the norms are already wrong (as ISSUES #12 demonstrated)."""
    records, _ = run_golden(tmp_path)
    norms = [r["grad_norm"] for r in records]
    assert norms == pytest.approx(GOLDEN_GRAD_NORMS, rel=1e-6), (
        f"gradient norms changed:\n  recorded {GOLDEN_GRAD_NORMS}\n  got      {norms}" + RERECORD
    )


def test_golden_learning_rate_schedule(tmp_path):
    records, _ = run_golden(tmp_path)
    assert [r["lr"] for r in records] == pytest.approx(GOLDEN_LRS, rel=1e-12)


def test_the_workload_is_reproducible_in_one_process(tmp_path):
    """Run twice, get the same thing. If this fails, the golden values are meaningless."""
    first, _ = run_golden(tmp_path / "a")
    second, _ = run_golden(tmp_path / "b")
    assert [r["loss"] for r in first] == [r["loss"] for r in second]


def test_bf16_moves_the_trajectory_only_slightly(tmp_path):
    """Not a golden value -- bf16 rounding is not worth pinning to 1e-6 -- but the trajectory must
    stay recognisably the same, which is the actual claim mixed precision makes."""
    records, _ = run_golden(tmp_path, precision="bf16")
    losses = [r["loss"] for r in records]
    assert losses == pytest.approx(GOLDEN_LOSSES, rel=2e-2)


# --- structural invariants ----------------------------------------------------------------------


@pytest.mark.parametrize("preset", sorted(PRESET_PARAMS))
def test_preset_parameter_counts_are_unchanged(preset):
    total, non_embedding = PRESET_PARAMS[preset]
    cfg = load_config(REPO_ROOT / "configs" / f"{preset}.yaml").model
    assert cfg.num_params() == total
    assert cfg.num_params(include_embeddings=False) == non_embedding


def test_state_dict_keys_are_unchanged():
    """A checkpoint written today must still load tomorrow. Renaming a module silently breaks
    every stored checkpoint, and nothing else in the suite would notice."""
    keys = sorted(GPT(ModelConfig(**NANO)).state_dict().keys())
    assert keys == NANO_STATE_DICT_KEYS, (
        "state_dict keys changed, which invalidates existing checkpoints" + RERECORD
    )


def test_tying_is_still_present_in_the_state_dict():
    """`lm_head.weight` and `tok_emb.weight` are the same tensor but both appear -- which is what
    makes a consolidated FSDP/TP checkpoint load into a plain model (D21)."""
    model = GPT(ModelConfig(**NANO))
    state = model.state_dict()
    assert "lm_head.weight" in state and "tok_emb.weight" in state
    assert torch.equal(state["lm_head.weight"], state["tok_emb.weight"])
    assert model.lm_head.weight is model.tok_emb.weight


def test_loss_at_init_is_still_about_ln_vocab(tmp_path):
    """Targets must be independent of the inputs, which is less obvious than it looks.

    With tied embeddings, "predict the token at this position" is already easy at initialization:
    the residual stream carries `tok_emb(x[t])`, and the tied LM head scores exactly that token
    highly. Passing `targets=idx` therefore measures ~4.46 rather than ln(128) = 4.85 -- a real
    property of weight tying, not a broken init.
    """
    import math

    fresh = Trainer(golden_experiment(tmp_path / "fresh"), out_dir=tmp_path / "fresh", echo=False)
    generator = torch.Generator().manual_seed(0)
    size = (4, 8)
    idx = torch.randint(0, NANO["vocab_size"], size, generator=generator)
    targets = torch.randint(0, NANO["vocab_size"], size, generator=generator)
    with torch.no_grad():
        _, loss = fresh.raw_model(idx, targets)
    assert loss.item() == pytest.approx(math.log(NANO["vocab_size"]), abs=0.3)


def test_every_config_in_the_repo_still_loads():
    """A config that stops parsing is a regression the rest of the suite would not catch, since
    the tests build their configs in Python."""
    configs = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    assert len(configs) >= 7  # tiny/small/medium, benchmark, and the three 2gpu strategy configs
    for path in configs:
        cfg = load_config(path)
        assert cfg.model.num_params() > 0
