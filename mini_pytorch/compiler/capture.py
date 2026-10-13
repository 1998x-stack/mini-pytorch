"""Dispatcher-level eager-to-graph capture for mini-pytorch v0.11."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .ir import Graph, TensorConstant, ValueRef, tensor_metadata


class GraphBreakError(RuntimeError):
    """Raised when eager dataflow leaves the supported capture boundary."""


@dataclass(frozen=True)
class CapturedGraph:
    graph: Graph
    eager_output: Any


_ACTIVE_CAPTURE: ContextVar["_CaptureSession | None"] = ContextVar(
    "mini_pytorch_active_capture", default=None
)


def is_capturing() -> bool:
    return _ACTIVE_CAPTURE.get() is not None


class _CaptureSession:
    def __init__(self, example_inputs: tuple["Tensor", ...]) -> None:
        self.graph = Graph()
        self._tensor_values: dict[int, int] = {}
        # Keep every observed Tensor alive for the full capture session.
        #
        # Python may recycle ``id(obj)`` immediately after a temporary object is
        # garbage-collected.  A bare id->Value map can therefore alias two
        # distinct eager temporaries in a long expression and assign multiple
        # producer nodes to one Graph Value.  Strong references make object-id
        # identity stable for the lifetime of this capture, while the explicit
        # ``is`` check below protects the invariant if bookkeeping ever changes.
        self._tensor_objects: dict[int, "Tensor"] = {}
        for tensor in example_inputs:
            value = self.graph.add_input(metadata=tensor_metadata(tensor))
            self._remember_tensor(tensor, value.id)

    def _lookup_tensor_value(self, tensor: "Tensor") -> int | None:
        key = id(tensor)
        known_object = self._tensor_objects.get(key)
        if known_object is None:
            return None
        if known_object is not tensor:
            raise RuntimeError(
                "capture tensor identity collision: Python object id was reused "
                "while the prior Tensor was still registered"
            )
        return self._tensor_values[key]

    def _remember_tensor(self, tensor: "Tensor", value_id: int) -> None:
        key = id(tensor)
        known_object = self._tensor_objects.get(key)
        if known_object is not None and known_object is not tensor:
            raise RuntimeError(
                "capture tensor identity collision: cannot bind two live Tensor "
                "objects to the same Python id"
            )
        self._tensor_objects[key] = tensor
        self._tensor_values[key] = int(value_id)

    def _tensor_ref(self, tensor: "Tensor", *, operator: str) -> ValueRef:
        known = self._lookup_tensor_value(tensor)
        if known is not None:
            return ValueRef(known)

        # A Tensor with a GradFn that was not produced by a captured Dispatcher
        # node came from unsupported eager dataflow (e.g. exp/log/layout op).
        # Silently treating it as a constant would produce a wrong graph.
        if tensor._grad_fn is not None:
            raise GraphBreakError(
                f"graph break before {operator}: encountered Tensor produced by an "
                "unsupported/unobserved eager operation; only Dispatcher-observed "
                "intermediates may flow between captured nodes in v0.11"
            )

        constant = TensorConstant(
            data=np.array(tensor.data, copy=True),
            requires_grad=bool(tensor.requires_grad),
            device=str(tensor.device),
        )
        value = self.graph.add_constant(
            constant,
            metadata=tensor_metadata(tensor),
        )
        self._remember_tensor(tensor, value.id)
        return ValueRef(value.id)

    def encode(self, value: Any, *, operator: str) -> Any:
        from ..tensor import Tensor

        if isinstance(value, Tensor):
            return self._tensor_ref(value, operator=operator)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            return tuple(self.encode(item, operator=operator) for item in value)
        if isinstance(value, list):
            return [self.encode(item, operator=operator) for item in value]
        if isinstance(value, dict):
            return {
                self.encode(key, operator=operator): self.encode(item, operator=operator)
                for key, item in value.items()
            }
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise GraphBreakError(
            f"graph break at {operator}: unsupported Python literal type "
            f"{type(value).__name__}"
        )

    def encode_output(self, value: Any, *, operator: str) -> Any:
        from ..tensor import Tensor

        if isinstance(value, Tensor):
            known = self._lookup_tensor_value(value)
            if known is not None:
                return ValueRef(known)
            out = self.graph.add_intermediate(metadata=tensor_metadata(value))
            self._remember_tensor(value, out.id)
            return ValueRef(out.id)
        if isinstance(value, tuple):
            return tuple(self.encode_output(x, operator=operator) for x in value)
        if isinstance(value, list):
            return [self.encode_output(x, operator=operator) for x in value]
        raise GraphBreakError(
            f"graph break at {operator}: Dispatcher output type "
            f"{type(value).__name__} is not supported by v0.11 capture"
        )

    def record_dispatch(self, operator: str, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
        arg_spec = self.encode(args, operator=operator)
        kwarg_spec = self.encode(kwargs, operator=operator)
        output_spec = self.encode_output(result, operator=operator)
        self.graph.add_node(operator, arg_spec, kwarg_spec, output_spec)

    def mark_graph_outputs(self, result: Any) -> None:
        # At function return, every Tensor must already be a graph input,
        # constant, or Dispatcher-produced intermediate. A direct unsupported
        # op result therefore raises a GraphBreak here as well.
        self.graph.outputs = self.encode(result, operator="<graph output>")
        self.graph.validate()


def _record_dispatch_if_active(operator: str, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
    session = _ACTIVE_CAPTURE.get()
    if session is not None:
        session.record_dispatch(operator, args, kwargs, result)


def capture(fn: Callable[..., Any], *example_inputs: "Tensor") -> CapturedGraph:
    """Execute ``fn`` eagerly while recording Dispatcher-visible operations.

    v0.11 intentionally requires positional Tensor graph inputs. General Python
    argument specialization and guards belong to v0.12.
    """

    from ..tensor import Tensor

    if not callable(fn):
        raise TypeError("capture expects a callable")
    if is_capturing():
        raise RuntimeError("nested graph capture is not supported in v0.11")
    if not example_inputs:
        raise ValueError("capture requires at least one Tensor example input")
    if not all(isinstance(value, Tensor) for value in example_inputs):
        raise TypeError("v0.11 capture inputs must all be Tensor objects")
    if len({id(value) for value in example_inputs}) != len(example_inputs):
        raise ValueError("the same Tensor object cannot occupy multiple graph input slots")

    session = _CaptureSession(tuple(example_inputs))
    token = _ACTIVE_CAPTURE.set(session)
    try:
        eager_output = fn(*example_inputs)
        session.mark_graph_outputs(eager_output)
    finally:
        _ACTIVE_CAPTURE.reset(token)

    return CapturedGraph(session.graph, eager_output)


try:  # pragma: no cover
    from ..tensor import Tensor
except Exception:  # pragma: no cover
    Tensor = Any
