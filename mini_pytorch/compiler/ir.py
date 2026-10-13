"""Small explicit graph IR used by mini-pytorch v0.11.

The IR records logical operator semantics, not backend kernels. Runtime Python
object identity is used only while a capture is active; persistent graph edges
are represented by deterministic integer Value ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ValueRef:
    """Reference to one graph Value inside structured node arguments."""

    value_id: int


@dataclass(frozen=True)
class TensorConstant:
    """Device-aware Tensor value captured by value for graph replay."""

    data: np.ndarray
    requires_grad: bool
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", np.array(self.data, copy=True))


@dataclass
class Value:
    id: int
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
    constant: TensorConstant | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"input", "constant", "intermediate"}:
            raise ValueError(f"unknown Value kind: {self.kind}")
        if self.kind == "constant" and self.constant is None:
            raise ValueError("constant Value requires TensorConstant payload")
        if self.kind != "constant" and self.constant is not None:
            raise ValueError("only constant Values may carry constant payload")


@dataclass
class Node:
    id: int
    operator: str
    args: Any
    kwargs: Any
    outputs: Any


class Graph:
    """Ordered SSA-like educational graph."""

    def __init__(self) -> None:
        self.values: dict[int, Value] = {}
        self.nodes: list[Node] = []
        self.inputs: list[int] = []
        self.outputs: Any = None
        self._next_value_id = 0
        self._next_node_id = 0

    def _new_value(
        self,
        kind: str,
        *,
        metadata: dict[str, Any] | None = None,
        constant: TensorConstant | None = None,
    ) -> Value:
        value = Value(
            id=self._next_value_id,
            kind=kind,
            metadata=dict(metadata or {}),
            constant=constant,
        )
        self.values[value.id] = value
        self._next_value_id += 1
        return value

    def add_input(self, *, metadata: dict[str, Any]) -> Value:
        value = self._new_value("input", metadata=metadata)
        self.inputs.append(value.id)
        return value

    def add_constant(
        self,
        constant: TensorConstant,
        *,
        metadata: dict[str, Any],
    ) -> Value:
        return self._new_value("constant", metadata=metadata, constant=constant)

    def add_intermediate(self, *, metadata: dict[str, Any]) -> Value:
        return self._new_value("intermediate", metadata=metadata)

    def add_node(self, operator: str, args: Any, kwargs: Any, outputs: Any) -> Node:
        node = Node(
            id=self._next_node_id,
            operator=str(operator),
            args=args,
            kwargs=kwargs,
            outputs=outputs,
        )
        self.nodes.append(node)
        self._next_node_id += 1
        return node

    def value(self, value_id: int) -> Value:
        try:
            return self.values[int(value_id)]
        except KeyError as exc:
            raise KeyError(f"unknown graph value %{value_id}") from exc

    def validate(self) -> None:
        for value_id in self.inputs:
            if self.value(value_id).kind != "input":
                raise ValueError(f"graph input %{value_id} is not an input Value")
        for node in self.nodes:
            _validate_refs(node.args, self.values)
            _validate_refs(node.kwargs, self.values)
            _validate_refs(node.outputs, self.values)
        _validate_refs(self.outputs, self.values)

    def format(self) -> str:
        """Return deterministic human-readable textual IR."""

        input_text = ", ".join(
            f"%v{vid}: {_format_metadata(self.value(vid).metadata)}"
            for vid in self.inputs
        )
        lines = [f"graph({input_text}):"]

        for vid in sorted(self.values):
            value = self.values[vid]
            if value.kind != "constant":
                continue
            payload = value.constant
            assert payload is not None
            data = np.array2string(
                payload.data,
                separator=", ",
                precision=8,
                threshold=16,
                edgeitems=3,
            ).replace("\n", " ")
            lines.append(
                f"  %v{vid} = constant_tensor("
                f"{_format_metadata(value.metadata)}, requires_grad={payload.requires_grad}, "
                f"data={data})"
            )

        for node in self.nodes:
            lhs = _format_spec(node.outputs)
            args = _format_spec(node.args)
            kwargs = _format_spec(node.kwargs)
            lines.append(
                f"  {lhs} = {node.operator}(args={args}, kwargs={kwargs})  # node{node.id}"
            )

        lines.append(f"  return {_format_spec(self.outputs)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def tensor_metadata(tensor: "Tensor") -> dict[str, Any]:
    return {
        "shape": tuple(int(x) for x in tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def _validate_refs(spec: Any, values: dict[int, Value]) -> None:
    if isinstance(spec, ValueRef):
        if spec.value_id not in values:
            raise ValueError(f"reference to missing graph value %{spec.value_id}")
        return
    if isinstance(spec, (tuple, list)):
        for item in spec:
            _validate_refs(item, values)
        return
    if isinstance(spec, dict):
        for key, value in spec.items():
            _validate_refs(key, values)
            _validate_refs(value, values)


def _format_metadata(metadata: dict[str, Any]) -> str:
    shape = metadata.get("shape")
    dtype = metadata.get("dtype")
    device = metadata.get("device")
    grad = metadata.get("requires_grad")
    return f"Tensor(shape={shape}, dtype={dtype}, device={device}, requires_grad={grad})"


def _format_spec(spec: Any) -> str:
    if isinstance(spec, ValueRef):
        return f"%v{spec.value_id}"
    if isinstance(spec, tuple):
        body = ", ".join(_format_spec(x) for x in spec)
        if len(spec) == 1:
            body += ","
        return f"({body})"
    if isinstance(spec, list):
        return "[" + ", ".join(_format_spec(x) for x in spec) + "]"
    if isinstance(spec, dict):
        items = sorted(spec.items(), key=lambda kv: repr(kv[0]))
        return "{" + ", ".join(
            f"{_format_spec(k)}: {_format_spec(v)}" for k, v in items
        ) + "}"
    return repr(spec)


try:  # pragma: no cover
    from ..tensor import Tensor
except Exception:  # pragma: no cover
    Tensor = Any
