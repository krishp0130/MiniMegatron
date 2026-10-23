# MiniMegatron: agent guide

Educational, engineering-quality distributed transformer training framework (GPT model, DDP,
FSDP2, tensor parallelism, Triton kernels, correctness and benchmark harness).

## Read first

1. `docs/PROGRESS.md`: current milestone, what's done, what's next, session log
2. `docs/ISSUES.md`: known problems, environment workarounds, design decisions
3. `docs/SCOPE.md`: full specification and milestone definitions (source of truth for scope)
4. `docs/milestones/M<n>_*.md`: detailed design for the active milestone (math, decisions, tests,
   exit criteria). Each milestone is scoped in its own doc before implementation starts.

## How this project is worked on

- The owner is building this to **learn** the material. Work one milestone step at a time,
  explain the concepts and design choices as you go, and check in before moving to the next step.
  Don't implement whole milestones in one pass.
- Order matters: correct baseline → DDP → FSDP2 → tensor parallelism → Triton. Don't start a
  milestone until the previous one's correctness tests pass.
- **Every optimization needs a reference implementation, a correctness test, and a reproducible
  benchmark.**
- **Never fabricate or estimate results.** GPU/NCCL/Triton work stays marked *pending* until
  actually run on NVIDIA hardware. Record measured numbers in PROGRESS.md with config, hardware,
  and commit.
- Prefer simple, readable code over abstraction. Keep dependencies minimal.

## Keeping context docs current (required)

At the end of each working session or milestone step:
- Update `docs/PROGRESS.md`: tick checklist items, update "Current state" and "Next up", and add
  a dated session-log entry (what was done, test results, anything left unfinished).
- Add any new problem or non-obvious workaround to `docs/ISSUES.md` (numbered, never renumbered),
  and any significant design decision as a `D<n>` entry.
- Scope changes go in `docs/SCOPE.md` plus its "Scope changes" table.
- Keep the README milestone table in sync.

## Environment

- Dev machine: Apple M4 Pro (CPU + MPS). **No CUDA, NCCL, or Triton locally** (ISSUES.md #3).
- Venv: `.venv/` (Python 3.12, `torch==2.14.1`). Use `.venv/bin/python`, `.venv/bin/pytest`,
  `.venv/bin/torchrun`.
- Environment report: `.venv/bin/python scripts/check_env.py`

## Commands

```bash
.venv/bin/pytest -q                       # all tests
.venv/bin/pytest tests/unit -q            # unit tests
.venv/bin/pip install -e ".[dev,text,logging]"

# multi-process on macOS: gloo needs loopback pinning (ISSUES.md #1)
GLOO_SOCKET_IFNAME=lo0 .venv/bin/torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29500 <script or -m module>
```

## Gotchas

- Without the loopback settings above, torchrun with gloo **hangs silently** on macOS. Always
  run distributed subprocesses with a hard timeout.
- FSDP2 / device meshes must be built with an explicit device type
  (`init_device_mesh("cpu", ...)`); auto-detection picks MPS and crashes (ISSUES.md #2).
- pytest markers: `gpu`, `multigpu`, `distributed`, `slow`. GPU tests must skip cleanly when
  CUDA is unavailable.

## Commits

Commit coherent units of work after tests pass, and push to `origin main`
(github.com/krishp0130/MiniMegatron).
