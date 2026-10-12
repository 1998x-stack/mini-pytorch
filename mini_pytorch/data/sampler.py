"""Sampling primitives inspired by ``torch.utils.data``.

The key design rule is separation of concerns:

* Dataset answers: "what is sample i?"
* Sampler answers: "which sample index comes next?"
* BatchSampler answers: "which indices form the next mini-batch?"
* DataLoader answers: "fetch those samples and collate them"

Keeping these responsibilities separate makes shuffle, custom sampling, class
balancing, distributed sampling, and batching extensible without changing the
Dataset API.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Sized

import numpy as np


class Sampler(ABC):
    """Base class that yields dataset indices.

    A sampler never loads data itself. It only defines index order.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Yield indices ``0, 1, ..., len(data_source)-1``."""

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Yield indices in random order, optionally with replacement.

    ``seed`` initializes an internal NumPy Generator once. Therefore two
    samplers created with the same seed produce the same sequence of epochs,
    while repeated iteration over one sampler still reshuffles every epoch.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        replacement: bool = False,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.data_source = data_source
        self.replacement = bool(replacement)
        self._num_samples = num_samples
        self._rng = np.random.default_rng(seed)

        if len(data_source) < 0:
            raise ValueError("data source length must be non-negative")
        if num_samples is not None and num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not self.replacement and num_samples not in {None, len(data_source)}:
            raise ValueError(
                "without replacement, num_samples must be omitted or equal to dataset length"
            )

    @property
    def num_samples(self) -> int:
        if self._num_samples is not None:
            return int(self._num_samples)
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            if n == 0 and self.num_samples > 0:
                raise ValueError("cannot sample from an empty dataset")
            indices = self._rng.integers(0, n, size=self.num_samples)
        else:
            indices = self._rng.permutation(n)
        return (int(index) for index in indices)

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler:
    """Wrap a sampler to yield lists of indices for mini-batches."""

    def __init__(self, sampler: Iterable[int], batch_size: int, drop_last: bool) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be bool")
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        for index in self.sampler:
            batch.append(int(index))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if not hasattr(self.sampler, "__len__"):
            raise TypeError("BatchSampler length requires a sized sampler")
        n = len(self.sampler)  # type: ignore[arg-type]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
