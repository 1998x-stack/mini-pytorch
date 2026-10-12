"""Autograd building blocks.

This module intentionally mirrors a small subset of ``torch.autograd.Function``:
operators implement ``forward`` and ``backward`` while ``Function.apply`` wires
those operators into a dynamic computation graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Type

import numpy as np


_GRAD_ENABLED = True


class no_grad:
    """Context manager that disables graph construction.

    Example:
        >>> with no_grad():
        ...     y = model(x)
    """

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

    def save_for_backward(self, *arrays: np.ndarray) -> None:
        self.saved_tensors = tuple(arrays)


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
    """Base class for differentiable operations.

    Subclasses provide two static methods:
      * ``forward(ctx, *raw_numpy_arrays, **kwargs)``
      * ``backward(ctx, grad_output)`` returning one gradient per Tensor input.
    """

    @classmethod
    def apply(cls, *inputs: "Tensor", **kwargs: Any) -> "Tensor":
        # Local import prevents a tensor <-> autograd import cycle.
        from .tensor import Tensor

        if not all(isinstance(x, Tensor) for x in inputs):
            raise TypeError("Function.apply expects Tensor positional inputs")

        ctx = Context()
        raw_inputs = [x.data for x in inputs]
        out_data = cls.forward(ctx, *raw_inputs, **kwargs)
        out_data = np.asarray(out_data)

        requires_grad = is_grad_enabled() and any(x.requires_grad for x in inputs)
        out = Tensor(out_data, requires_grad=requires_grad)
        if requires_grad:
            out._grad_fn = GradFn(function=cls, parents=tuple(inputs), ctx=ctx)
        return out

    @staticmethod
    def forward(ctx: Context, *args: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        raise NotImplementedError


# Type-checking only; imported lazily at runtime.
try:  # pragma: no cover
    from .tensor import Tensor
except Exception:  # pragma: no cover
    Tensor = Any
