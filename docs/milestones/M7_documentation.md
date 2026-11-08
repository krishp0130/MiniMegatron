# Milestone 7 — Final Documentation

**Goal:** make the repository runnable and judgeable by someone who did not build it. Six
milestones produced working code and a lot of recorded reasoning, but that reasoning lives in
milestone design docs and a 600-line ISSUES log — the right places to *write* it, the wrong places
to *read* it first.

**Not a rewrite.** Nearly all the substance exists. M7 organises it into the four documents
SCOPE.md §11 names, fills the results tables with what has actually been measured, and states the
limitations plainly instead of leaving a reader to infer them.

## Deliverables

| Document | Covers | Source material |
|---|---|---|
| `architecture.md` | the model, config system, training loop, layering, and *where the seams are and why* | M1 design doc, D4/D6/D8/D12 |
| `distributed_training.md` | the three strategies side by side; DDP in full (it has no standalone doc) | M2 design doc, D16/D17/D24 |
| `testing.md` | the six suites, the reference discipline, mutation testing and what it caught | all milestones, ISSUES #6/#7/#9/#11/#12/#13/#14/#15 |
| `benchmarks.md` | the method, how to reproduce, measured results, pending measurements with exact commands | M6 design doc, D27/D28 |

`fsdp.md` and `tensor_parallel.md` already exist as deep dives and are linked rather than
duplicated.

Plus, in the README: installation, a documentation index, **known limitations and future work**,
and an honest assessment against SCOPE.md §13's Definition of Done — including the items that
cannot be met without GPU hardware.

## The one rule for this milestone

**No claim may exceed its evidence.** Six milestones of discipline about this get undone by one
summary document that rounds "written but never executed" up to "implemented", or quotes a CPU
throughput figure without saying it is CPU. Specifically:

- The Triton kernel is written and **unverified** (ISSUES #16), everywhere it is mentioned.
- CPU benchmark numbers are labelled as CPU, and as relative overhead rather than scaling.
- Definition-of-Done items 3, 4, 5 and 7 need NVIDIA hardware and are marked **pending**, not done.
- Every pending measurement is listed with the command that would produce it, so the gap is a
  task rather than a mystery.

## Exit criteria

1. The four documents exist, and each links to code rather than restating it.
2. A reader can install, train, test and benchmark from the README alone.
3. Every pending item names the hardware it needs and the command that closes it.
4. The Definition of Done is assessed honestly, item by item.
5. `pytest -q` still passes; documentation changes break nothing.
