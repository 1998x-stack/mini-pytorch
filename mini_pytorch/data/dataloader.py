"""Single-process DataLoader built from Dataset + Sampler + BatchSampler."""
from __future__ import annotations

from typing import Callable, Iterable, Iterator

import numpy as np

from ..tensor import Tensor
from .dataset import Dataset
from .sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler


def _require_homogeneous(batch, expected_type, *, label: str) -> None:
    if not all(isinstance(item, expected_type) for item in batch):
        raise TypeError(f"default_collate requires homogeneous {label} samples")


def default_collate(batch):
    """Stack a homogeneous batch into Tensor objects.

    Collation is intentionally separate from sampling: samplers choose indices,
    while ``default_collate`` only converts already-fetched samples into a batch.
    Nested tuple/list samples must have equal lengths; silently truncating ragged
    samples would lose user data and is therefore rejected explicitly.
    """
    if not batch:
        raise ValueError("cannot collate an empty batch")

    first = batch[0]
    if isinstance(first, Tensor):
        _require_homogeneous(batch, Tensor, label="Tensor")
        return Tensor(np.stack([item.data for item in batch], axis=0))
    if isinstance(first, np.ndarray):
        _require_homogeneous(batch, np.ndarray, label="ndarray")
        return Tensor(np.stack(batch, axis=0))
    if isinstance(first, (int, float, np.number, bool)):
        scalar_types = (int, float, np.number, bool)
        _require_homogeneous(batch, scalar_types, label="scalar")
        return Tensor(np.asarray(batch))
    if isinstance(first, tuple):
        _require_homogeneous(batch, tuple, label="tuple")
        expected_len = len(first)
        if any(len(item) != expected_len for item in batch):
            raise ValueError("default_collate requires equal-size tuple samples")
        return tuple(default_collate(list(items)) for items in zip(*batch))
    if isinstance(first, list):
        _require_homogeneous(batch, list, label="list")
        expected_len = len(first)
        if any(len(item) != expected_len for item in batch):
            raise ValueError("default_collate requires equal-size list samples")
        return [default_collate(list(items)) for items in zip(*batch)]
    raise TypeError(f"default_collate does not support samples of type {type(first)!r}")


class DataLoader:
    """Iterate over a map-style Dataset in mini-batches.

    The important v0.3 architecture mirrors the conceptual PyTorch pipeline::

        Dataset <- indices -- Sampler -- BatchSampler -- DataLoader -- collate_fn

    ``shuffle=True`` is only convenience syntax for creating a ``RandomSampler``.
    A custom ``sampler`` lets callers define sample order without changing the
    dataset, and a custom ``batch_sampler`` controls complete batches.

    This version remains deliberately single-process; multiprocessing workers,
    prefetching, pinned memory, and distributed samplers are future extensions.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        *,
        shuffle: bool = False,
        sampler: Sampler | None = None,
        batch_sampler: Iterable[Iterable[int]] | None = None,
        drop_last: bool = False,
        seed: int | None = None,
        collate_fn: Callable | None = None,
    ) -> None:
        self.dataset = dataset
        self.collate_fn = collate_fn or default_collate

        if batch_sampler is not None:
            # Match the central PyTorch rule: a complete batch sampler owns
            # batching, so automatic batching options must not also be active.
            if sampler is not None:
                raise ValueError("sampler and batch_sampler are mutually exclusive")
            if shuffle:
                raise ValueError("shuffle and batch_sampler are mutually exclusive")
            if drop_last:
                raise ValueError("drop_last is controlled by batch_sampler")
            if batch_size != 1:
                raise ValueError("batch_size is controlled by batch_sampler")

            self.sampler = getattr(batch_sampler, "sampler", None)
            self.batch_sampler = batch_sampler
            self.batch_size = getattr(batch_sampler, "batch_size", None)
            self.drop_last = getattr(batch_sampler, "drop_last", False)
            self.shuffle = False
            return

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sampler is not None and shuffle:
            raise ValueError("sampler and shuffle=True are mutually exclusive")

        if sampler is None:
            sampler = (
                RandomSampler(dataset, seed=seed)
                if shuffle
                else SequentialSampler(dataset)
            )

        self.sampler = sampler
        self.batch_sampler = BatchSampler(sampler, batch_size, bool(drop_last))
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)

    def __len__(self) -> int:
        if not hasattr(self.batch_sampler, "__len__"):
            raise TypeError("DataLoader length requires a sized batch_sampler")
        return len(self.batch_sampler)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator:
        for batch_indices in self.batch_sampler:
            indices = [int(index) for index in batch_indices]
            if not indices:
                raise ValueError("batch_sampler yielded an empty batch")
            samples = [self.dataset[index] for index in indices]
            yield self.collate_fn(samples)
