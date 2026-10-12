"""Autograd graph-building primitives for mini-pytorch v0.8.

The module owns graph construction objects (`Context`, `GradFn`, `Function`) and
grad-mode state. Backward execution/lifetime policy lives in
``mini_pytorch.autograd_engine``.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Type
import weakref

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


class RemovableHandle:
    """Small idempotent handle returned by ``Tensor.register_hook``."""

    _next_id = itertools.count()

    def __init__(self, tensor: "Tensor") -> None:
        self.id = next(self._next_id)
        self._tensor_ref = weakref.ref(tensor)

    def remove(self) -> None:
        tensor = self._tensor_ref()
        if tensor is not None:
            tensor._backward_hooks.pop(self.id, None)


class Context:
    """Per-operation state passed from forward to backward.

    v0.8 makes lifetime explicit. A default backward releases Context state;
    ``retain_graph=True`` asks the Engine to preserve it for another execution.
    """

    def __init__(self) -> None:
        self.saved_tensors: tuple[np.ndarray, ...] = ()
        self._tensor_inputs: tuple["Tensor", ...] = ()
        self._raw_inputs: tuple[np.ndarray, ...] = ()
        self._saved_versions: list[tuple["Tensor", int]] = []
        self._released = False

    @property
    def is_released(self) -> bool:
        return self._released

    def ensure_available(self) -> None:
        if self._released:
            raise RuntimeError(
                "trying to backward through a graph whose saved state has already "
                "been released; call backward(retain_graph=True) on the earlier "
                "backward if the graph must be reused"
            )

    def _bind_inputs(
        self,
        tensors: tuple["Tensor", ...],
        arrays: tuple[np.ndarray, ...],
    ) -> None:
        self.ensure_available()
        self._tensor_inputs = tensors
        self._raw_inputs = arrays

    def save_for_backward(self, *arrays: np.ndarray) -> None:
        self.ensure_available()
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
        self.ensure_available()
        for tensor, expected in self._saved_versions:
            actual = tensor._version
            if actual != expected:
                raise RuntimeError(
                    "one of the tensors needed for gradient computation was "
                    "modified by an inplace operation: expected storage version "
                    f"{expected}, found {actual}"
                )

    def release(self) -> None:
        """Release references that exist only to execute backward.

        Shape/config attributes deliberately remain for inspectability, but the
        potentially large arrays/Tensor references are dropped and the Context
        becomes permanently unavailable for another backward execution.
        """

        if self._released:
            return
        self.saved_tensors = ()
        self._saved_versions = []
        self._tensor_inputs = ()
        self._raw_inputs = ()
        self._released = True


@dataclass
class GradFn:
    """One node of the dynamic reverse-mode graph."""

    function: Type["Function"]
    parents: tuple["Tensor", ...]
    ctx: Context

    @property
    def name(self) -> str:
        return f"{self.function.__name__}Backward"

    @property
    def is_released(self) -> bool:
        return self.ctx.is_released

    def release(self) -> None:
        self.ctx.release()


class Function:
    """Base class for differentiable educational operations.

    Subclasses implement numerical ``forward`` and vector-Jacobian-product
    ``backward`` methods. ``apply`` builds the GradFn/Context edge.
    """

    @classmethod
    def apply(cls, *inputs: "Tensor", **kwargs: Any) -> "Tensor":
        from .tensor import Tensor

        if not all(isinstance(x, Tensor) for x in inputs):
            raise TypeError("Function.apply expects Tensor positional inputs")

        devices = {x.device for x in inputs}
        if len(devices) > 1:
            rendered = ", ".join(sorted(str(device) for device in devices))
            raise ValueError(f"Function.apply received tensors on different devices: {rendered}")

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
