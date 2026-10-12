"""Backend ownership and allocation registry for mini-pytorch v0.7."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .device import Device, as_device


class Backend(ABC):
    """Owns allocation/copy policy for one device type."""

    device_type: str

    @abstractmethod
    def allocate(self, size: int, dtype) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def from_numpy(self, array: np.ndarray, *, copy: bool = True) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self, allocation: np.ndarray, *, copy: bool = False) -> np.ndarray:
        raise NotImplementedError


class NumPyBackend(Backend):
    device_type = "cpu"

    def allocate(self, size: int, dtype) -> np.ndarray:
        return np.empty(int(size), dtype=dtype)

    def from_numpy(self, array: np.ndarray, *, copy: bool = True) -> np.ndarray:
        array = np.asarray(array)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        flat = array.reshape(-1)
        return flat.copy() if copy else flat

    def to_numpy(self, allocation: np.ndarray, *, copy: bool = False) -> np.ndarray:
        return allocation.copy() if copy else allocation


class EducationalBackend(NumPyBackend):
    """A second backend identity backed by host NumPy memory.

    It exists so device-aware allocation/dispatch can be tested without GPU
    hardware. Numerical representation intentionally stays simple.
    """

    device_type = "edu"


_BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    if not isinstance(backend, Backend):
        raise TypeError("backend must implement Backend")
    if backend.device_type in _BACKENDS:
        raise ValueError(f"backend already registered for {backend.device_type!r}")
    _BACKENDS[backend.device_type] = backend


def backend_for(device: Device | str) -> Backend:
    resolved = as_device(device)
    try:
        return _BACKENDS[resolved.type]
    except KeyError as exc:
        raise ValueError(f"no backend registered for device {resolved}") from exc


def list_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


register_backend(NumPyBackend())
register_backend(EducationalBackend())
