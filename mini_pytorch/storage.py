"""Explicit tensor storage and layout helpers for mini-pytorch v0.4.

A Storage owns a flat contiguous allocation. Tensor objects may share one Storage
while exposing different shapes, strides and storage offsets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class VersionCounter:
    """Shared mutation counter for every Tensor aliasing the same Storage."""

    value: int = 0

    def bump(self) -> int:
        self.value += 1
        return self.value


def contiguous_strides(shape: Iterable[int]) -> tuple[int, ...]:
    """Return row-major/C-contiguous strides measured in elements."""
    shape = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in shape):
        raise ValueError("shape dimensions must be non-negative")
    strides = [0] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = running
        running *= max(shape[index], 1)
    return tuple(strides)


class Storage:
    """A flat contiguous allocation shared by one or more Tensor views."""

    def __init__(self, data, *, copy: bool = True) -> None:
        array = np.asarray(data)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        flat = array.reshape(-1)
        self._array = flat.copy() if copy else flat
        self._version_counter = VersionCounter()

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def size(self) -> int:
        return int(self._array.size)

    @property
    def nbytes(self) -> int:
        return int(self._array.nbytes)

    @property
    def version(self) -> int:
        return self._version_counter.value

    def bump_version(self) -> int:
        return self._version_counter.bump()

    def numpy(self) -> np.ndarray:
        """Return the flat allocation. Mutating this bypasses version tracking."""
        return self._array

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"Storage(size={self.size}, dtype={self.dtype}, "
            f"version={self.version})"
        )
