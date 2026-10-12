"""Autograd building blocks.

v0.4 adds saved-tensor version checks: when backward depends on an input value,
the Context remembers that Tensor's shared Storage version. Mutating any alias
through a tracked inplace Tensor operation invalidates backward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import numpy as np


_GRAD_ENABLED = True


class no_grad:
    """Context manager that disables graph construction."""

    def __enter__(self):
        global _GRAD_ENABLED
        self._previous = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, exc_type, exc, tb):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._previous
        return False


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


class Context:
    """Per-operation storage used to pass values from forward to backward."""

    def __init__(self) -> None:
        self.saved_tensors: tuple[np.ndarray, ...] = ()
        self._tensor_inputs: tuple["Tensor", ...] = ()
        self._raw_inputs: tuple[np.ndarray, ...] = ()
        self._saved_versions: list[tuple["Tensor", int]] = []

    def _bind_inputs(
        self,
        tensors: tuple["Tensor", ...],
        arrays: tuple[np.ndarray, ...],
    ) -> None:
        self._tensor_inputs = tensors
        self._raw_inputs = arrays

    def save_for_backward(self, *arrays: np.ndarray) -> None:
        self.saved_tensors = tuple(arrays)
        self._saved_versions = []
        seen_storage_ids: set[int] = set()
        for saved in arrays:
            if not isinstance(saved, np.ndarray):
                continue
            for tensor, raw in zip(self._tensor_inputs, self._raw_inputs):
                if saved is raw or np.shares_memory(saved, raw):
                    storage_id = id(tensor._storage)
                    if storage_id not in seen_storage_ids:
                        self._saved_versions.append((tensor, tensor._version))
                        seen_storage_ids.add(storage_id)
                    break

    def check_saved_versions(self) -> None:
        for tensor, expected in self._saved_versions:
            actual = tensor._version
            if actual != expected:
                raise RuntimeError(
                    "one of the tensors needed for gradient computation was "
                    "modified by an inplace operation: expected storage version "
                    f"{expected}, found {actual}"
                )


@dataclass
class GradFn:
    """One node of the dynamic autograd graph."""

    function: Type["Function"]
    parents: tuple["Tensor", ...]
    ctx: Context

    @property
    def name(self) -> str:
        return f"{self.function.__name__}Backward"


class Function:
    """Base class for differentiable operations."""

    @classmethod
    def apply(cls, *inputs: "Tensor", **kwargs: Any) -> "Tensor":
        from .tensor import Tensor

        if not all(isinstance(x, Tensor) for x in inputs):
            raise TypeError("Function.apply expects Tensor positional inputs")

        ctx = Context()
        raw_inputs = tuple(x.data for x in inputs)
        ctx._bind_inputs(tuple(inputs), raw_inputs)
        out_data = np.asarray(cls.forward(ctx, *raw_inputs, **kwargs))

        requires_grad = is_grad_enabled() and any(x.requires_grad for x in inputs)
        out = Tensor._from_operation_result(
            out_data,
            inputs=tuple(inputs),
            requires_grad=requires_grad,
        )
        if requires_grad:
            out._grad_fn = GradFn(function=cls, parents=tuple(inputs), ctx=ctx)
        return out

    @staticmethod
    def forward(ctx: Context, *args: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        raise NotImplementedError


try:  # pragma: no cover
    from .tensor import Tensor
except Exception:  # pragma: no cover
    Tensor = Any
