"""Synthetic datasets and deterministic batching.

Two properties are designed in here, and everything later depends on them (D9):

1. **Index determinism.** Sample *i* is a pure function of `(seed, i)`. Nothing is stored, so any
   sampler -- including Milestone 2's `DistributedSampler` -- gets reproducible shards.
2. **A batch-size-independent index stream.** The order samples are visited in is a pure function
   of `(seed, sample_offset)` and does *not* depend on the batch size. So `micro_batch_size=2`
   with 4 accumulation steps visits exactly the same samples, in the same order, as
   `micro_batch_size=8` with 1 -- which is what makes the gradient-accumulation equivalence test
   (and Milestone 2's DDP equivalence test) exact rather than approximate. It also means
   "resume at step s" needs no dataloader state saved at all.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _mix(seed: int, index: int) -> int:
    """SplitMix64's finalizer: turn a nearby (seed, index) pair into a well-separated seed.

    Seeding a generator with `seed + index` would give neighbouring items highly correlated
    streams. This avalanches every input bit across all 64 output bits first.
    """
    z = (seed + index * 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) >> 1  # >> 1 keeps it inside the positive int64 range


def _generator(seed: int, index: int) -> torch.Generator:
    g = torch.Generator()  # always CPU: data generation must not depend on the training device
    g.manual_seed(_mix(seed, index))
    return g


class RandomTokenDataset(Dataset):
    """Uniform random tokens. Item *i* is a pure function of `(seed, i)`.

    Nothing is learnable here -- loss stays at ln(vocab_size) -- which is exactly what a
    throughput benchmark wants: no I/O, no preprocessing, and no learning curve to confuse the
    measurement.
    """

    def __init__(self, vocab_size: int, seq_len: int, n_items: int, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_items = n_items
        self.seed = seed

    def __len__(self) -> int:
        return self.n_items

    def _sequence_index(self, index: int) -> int:
        return index

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < self.n_items:
            raise IndexError(f"index {index} out of range for {self.n_items} items")
        # seq_len + 1 tokens, so the inputs and the targets shifted one step both have seq_len.
        tokens = torch.randint(
            0,
            self.vocab_size,
            (self.seq_len + 1,),
            generator=_generator(self.seed, self._sequence_index(index)),
            dtype=torch.int64,
        )
        return tokens[:-1], tokens[1:]  # x, y: position t's target is the token at t+1


class MemorizationDataset(RandomTokenDataset):
    """`n_sequences` fixed sequences, repeated. The model can memorize them, so loss -> 0.

    This is what makes the overfit test a real proof: a model that cannot drive loss to ~0 on
    data it has seen hundreds of times has something broken in the forward, the backward, the
    optimizer, or the target alignment.
    """

    def __init__(
        self, vocab_size: int, seq_len: int, n_items: int, n_sequences: int, seed: int = 0
    ) -> None:
        super().__init__(vocab_size, seq_len, n_items, seed)
        self.n_sequences = n_sequences

    def _sequence_index(self, index: int) -> int:
        return index % self.n_sequences  # the only difference: the generator seed repeats


class IndexStream:
    """An infinite deterministic stream of dataset indices.

    The stream is the concatenation of one seeded permutation per epoch, so every item is seen
    once per epoch, and `take(offset, count)` is a pure function of `(seed, offset, count)` --
    independent of how the stream is later chopped into batches.
    """

    def __init__(self, n_items: int, seed: int = 0) -> None:
        if n_items <= 0:
            raise ValueError(f"n_items must be positive, got {n_items}")
        self.n_items = n_items
        self.seed = seed
        self._cache: dict[int, torch.Tensor] = {}

    def permutation(self, epoch: int) -> torch.Tensor:
        if epoch not in self._cache:
            self._cache[epoch] = torch.randperm(
                self.n_items, generator=_generator(self.seed, epoch)
            )
            # Reads are sequential, so two epochs are all that a wrapping `take` ever needs.
            if len(self._cache) > 2:
                self._cache.pop(next(iter(self._cache)))  # dicts keep insertion order
        return self._cache[epoch]

    def take(self, offset: int, count: int) -> torch.Tensor:
        """`count` indices starting at stream position `offset`, wrapping across epochs."""
        if offset < 0 or count < 0:
            raise ValueError(f"offset and count must be non-negative, got {offset}, {count}")
        chunks = []
        taken = 0
        while taken < count:
            pos = offset + taken
            epoch, within = divmod(pos, self.n_items)
            chunk = self.permutation(epoch)[within : within + (count - taken)]
            chunks.append(chunk)
            taken += len(chunk)
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]


class BatchStream:
    """Turns an `IndexStream` into micro-batches of `(x, y)` tensors.

    Micro-batch *m* holds stream positions `[m * micro_batch_size, (m+1) * micro_batch_size)`.
    The trainer's micro-batch index is `step * grad_accum_steps + j`, so the samples consumed up
    to any step depend only on the seed and the number of samples consumed, never on how they
    were grouped.
    """

    def __init__(self, dataset: Dataset, micro_batch_size: int, seed: int = 0) -> None:
        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
        self.dataset = dataset
        self.micro_batch_size = micro_batch_size
        self.stream = IndexStream(len(dataset), seed)

    def indices(self, micro_index: int) -> torch.Tensor:
        return self.stream.take(micro_index * self.micro_batch_size, self.micro_batch_size)

    def micro_batch(self, micro_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        samples = [self.dataset[int(i)] for i in self.indices(micro_index)]
        xs, ys = zip(*samples)
        return torch.stack(xs), torch.stack(ys)


def build_dataset(data_cfg, model_cfg, split: str = "train") -> RandomTokenDataset:
    """Build the configured dataset for a split. `val` differs only by seed: a disjoint split."""
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    seed = data_cfg.seed if split == "train" else data_cfg.val_seed

    if data_cfg.dataset == "memorization":
        return MemorizationDataset(
            vocab_size=model_cfg.vocab_size,
            seq_len=model_cfg.seq_len,
            n_items=data_cfg.n_items,
            n_sequences=data_cfg.n_sequences,
            seed=seed,
        )
    return RandomTokenDataset(
        vocab_size=model_cfg.vocab_size,
        seq_len=model_cfg.seq_len,
        n_items=data_cfg.n_items,
        seed=seed,
    )
