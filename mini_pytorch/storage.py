"""Explicit tensor storage, layout helpers and backend identity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .backend import Backend, backend_for
from .device import Device, as_device


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
    """A flat backend-owned allocation shared by one or more Tensor views."""

    def __init__(
        self,
        data,
        *,
        copy: bool = True,
        device: Device | str = "cpu",
    ) -> None:
        self._device = as_device(device)
        self._backend: Backend = backend_for(self._device)
        array = np.asarray(data)
        self._array = self._backend.from_numpy(array, copy=copy)
        self._version_counter = VersionCounter()

    @classmethod
    def empty(cls, size: int, dtype, *, device: Device | str = "cpu") -> "Storage":
        obj = cls.__new__(cls)
        obj._device = as_device(device)
        obj._backend = backend_for(obj._device)
        obj._array = obj._backend.allocate(int(size), dtype)
        obj._version_counter = VersionCounter()
        return obj

    @property
    def device(self) -> Device:
        return self._device

    @property
    def backend(self) -> Backend:
        return self._backend

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

    def numpy(self, *, copy: bool = False) -> np.ndarray:
        """Expose the flat allocation through the backend's NumPy bridge.

        Mutating a non-copy result bypasses version tracking and is therefore a
        low-level educational escape hatch.
        """

        return self._backend.to_numpy(self._array, copy=copy)

    def clone_to(self, device: Device | str) -> "Storage":
        target = as_device(device)
        logical = self._backend.to_numpy(self._array, copy=True)
        return Storage(logical, copy=True, device=target)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"Storage(size={self.size}, dtype={self.dtype}, device={self.device}, "
            f"version={self.version})"
        )
