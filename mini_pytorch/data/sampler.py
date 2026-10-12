"""Sampling primitives inspired by ``torch.utils.data``.

The key design rule is separation of concerns:

* Dataset answers: "what is sample i?"
* Sampler answers: "which sample index comes next?"
* BatchSampler answers: "which indices form the next mini-batch?"
* DataLoader answers: "fetch those samples and collate them"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Sized

import numpy as np

from ..runtime.rng import integers as framework_integers
from ..runtime.rng import permutation as framework_permutation


class Sampler(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Yield indices randomly, with framework or explicitly private RNG.

    ``seed=None`` participates in mini-pytorch's checkpointable framework RNG.
    An explicit seed preserves the historical component-local stream semantics.
    v0.9 guarantees exact checkpoint resume at documented training boundaries;
    arbitrary partially-consumed iterator state is intentionally out of scope.
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
        self.seed = seed
        self._rng = None if seed is None else np.random.default_rng(seed)

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
            if self._rng is None:
                indices = framework_integers(0, n, size=self.num_samples)
            else:
                indices = self._rng.integers(0, n, size=self.num_samples)
        else:
            if self._rng is None:
                indices = framework_permutation(n)
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
