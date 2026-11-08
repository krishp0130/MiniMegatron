"""The real launch path: `torchrun`, two processes, end to end.

The spawn-based harness exercises our code; this exercises the way a user actually starts a job,
including the macOS gloo workaround that only applies to a real launch (ISSUES #1).
"""

import json
import os
import subprocess
import sys

import pytest

from tests.distributed.conftest import LOOPBACK_IFNAME, free_port

pytestmark = [pytest.mark.distributed, pytest.mark.slow]

TIMEOUT = 300


def torchrun(*module_args: str, nproc: int = 2) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc_per_node={nproc}",
            "--master_addr=127.0.0.1",
            f"--master_port={free_port()}",
            *module_args,
        ],
        # gloo otherwise binds an interface the peers can't reach and hangs with no error.
        env={**os.environ, "GLOO_SOCKET_IFNAME": LOOPBACK_IFNAME},
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def write_config(path, out_dir) -> None:
    path.write_text(
        "model:\n"
        "  vocab_size: 128\n  seq_len: 16\n  n_layers: 2\n  n_heads: 4\n  d_model: 32\n"
        "  d_ff: 64\n  dropout: 0.0\n"
        "data:\n"
        "  dataset: memorization\n  n_items: 256\n  n_sequences: 16\n  seed: 0\n"
        "train:\n"
        "  max_steps: 4\n  micro_batch_size: 4\n  grad_accum_steps: 1\n"
        "  global_batch_size: 8\n"  # 4 x 1 x 2 ranks: validates the world size is seen
        "  lr: 1.0e-3\n  device: cpu\n  log_interval: 2\n"
        f"  out_dir: {out_dir}\n"
        "distributed:\n  backend: gloo\n"
    )


def test_torchrun_trains_and_all_ranks_exit_cleanly(tmp_path):
    """Exit criterion 6."""
    config, out_dir = tmp_path / "nano.yaml", tmp_path / "run"
    write_config(config, out_dir)

    result = torchrun("-m", "minimegatron.train", "--config", str(config))
    assert result.returncode == 0, result.stderr  # torchrun propagates any rank's failure

    assert "ranks: 2" in result.stdout
    assert result.stdout.count("non-embedding") == 1  # rank 0 printed the summary, alone

    lines = [json.loads(line) for line in (out_dir / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [2, 4]
    assert lines[-1]["tokens_seen"] == 4 * 4 * 2 * 16  # steps x micro x ranks x seq_len
    assert sorted(p.name for p in out_dir.iterdir()) == ["ckpt_000004.pt", "log.jsonl"]


def test_torchrun_rejects_a_global_batch_size_that_ignores_the_world_size(tmp_path):
    """`global_batch_size` must be checked against micro x accum x *ranks*, not micro x accum."""
    config, out_dir = tmp_path / "nano.yaml", tmp_path / "run"
    write_config(config, out_dir)

    result = torchrun(
        "-m", "minimegatron.train", "--config", str(config),
        "--set", "train.global_batch_size=4",  # correct for 1 rank, wrong for 2
    )
    assert result.returncode != 0
    assert "global_batch_size 4" in result.stderr or "global_batch_size 4" in result.stdout


def test_collectives_demo_runs_under_torchrun(tmp_path):
    result = torchrun("-m", "minimegatron.distributed.collectives_demo")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("grad norm local") == 2  # one line per rank
    assert "That is the whole of data parallelism" in result.stdout


def test_collectives_demo_explains_itself_when_run_alone():
    """A single process can't demonstrate a collective; say so instead of crashing."""
    result = subprocess.run(
        [sys.executable, "-m", "minimegatron.distributed.collectives_demo"],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    assert result.returncode == 1
    assert "needs at least 2 processes" in result.stdout
