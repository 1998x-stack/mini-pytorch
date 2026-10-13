"""DistributedSampler: deterministic dataset-index partitioning by rank."""
from __future__ import annotations

import math
from typing import Iterator, Sized

import numpy as np

from ..data.sampler import Sampler


class DistributedSampler(Sampler):
    """Partition one global map-style dataset across replicated ranks.

    The global ordering is created first, then every rank takes a strided slice::

        global_indices[rank:total_size:num_replicas]

    ``set_epoch`` changes the deterministic shuffled ordering without coupling the
    sampler to the framework checkpoint RNG.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if not isinstance(num_replicas, (int, np.integer)) or int(num_replicas) <= 0:
            raise ValueError("num_replicas must be a positive integer")
        self.num_replicas = int(num_replicas)
        if not isinstance(rank, (int, np.integer)) or not 0 <= int(rank) < self.num_replicas:
            raise ValueError("rank must satisfy 0 <= rank < num_replicas")
        self.rank = int(rank)
        self.data_source = data_source
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        self.seed = int(seed)
        self.epoch = 0

        size = len(data_source)
        if size < 0:
            raise ValueError("data source length must be non-negative")
        if self.drop_last:
            self.total_size = (size // self.num_replicas) * self.num_replicas
            self.num_samples = self.total_size // self.num_replicas
        else:
            self.num_samples = int(math.ceil(size / self.num_replicas)) if size else 0
            self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, (int, np.integer)):
            raise TypeError("epoch must be an integer")
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        size = len(self.data_source)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            indices = [int(i) for i in rng.permutation(size)]
        else:
            indices = list(range(size))

        if self.drop_last:
            indices = indices[: self.total_size]
        elif self.total_size > len(indices):
            padding = self.total_size - len(indices)
            if not indices:
                return iter(())
            repeats = (padding + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:padding])

        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError(
                f"internal DistributedSampler partition error: expected {self.num_samples}, got {len(rank_indices)}"
            )
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples
