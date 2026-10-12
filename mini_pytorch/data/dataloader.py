"""Single-process DataLoader and batching utilities."""
from __future__ import annotations

import math
from typing import Callable, Iterator

import numpy as np

from ..tensor import Tensor
from .dataset import Dataset


def default_collate(batch):
    """Stack a homogeneous batch into Tensor objects."""
    if not batch:
        raise ValueError("cannot collate an empty batch")

    first = batch[0]
    if isinstance(first, Tensor):
        return Tensor(np.stack([item.data for item in batch], axis=0))
    if isinstance(first, np.ndarray):
        return Tensor(np.stack(batch, axis=0))
    if isinstance(first, (int, float, np.number, bool)):
        return Tensor(np.asarray(batch))
    if isinstance(first, tuple):
        return tuple(default_collate(list(items)) for items in zip(*batch))
    if isinstance(first, list):
        return [default_collate(list(items)) for items in zip(*batch)]
    raise TypeError(f"default_collate does not support samples of type {type(first)!r}")


class DataLoader:
    """Iterate over a Dataset in mini-batches.

    This first implementation is deliberately single-process. Its public boundary
    leaves room for Sampler/workers/pinned-memory concepts in later versions.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
        collate_fn: Callable | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.collate_fn = collate_fn or default_collate
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            samples = [self.dataset[int(index)] for index in batch_indices]
            yield self.collate_fn(samples)
