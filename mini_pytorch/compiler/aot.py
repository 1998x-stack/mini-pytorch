"""Educational AOTAutograd-like graph differentiation for mini-pytorch v0.13.

The implementation deliberately supports a small logical-op subset. It turns
reverse-mode VJP execution into a second explicit Graph while keeping the eager
AutogradEngine as the numerical/correctness reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..autograd import no_grad
from ..tensor import Tensor
from .capture import capture
from .interpreter import GraphInterpreter
from .ir import Graph, Node, ValueRef


class AOTDifferentiationError(RuntimeError):
    """Raised when a forward Graph cannot be differentiated by v0.13."""


_SUPPORTED = {
    "aten::add.Tensor",
    "aten::mul.Tensor",
    "aten::matmul.default",
    "aten::relu.default",
    "aten::sum.default",
}


@dataclass(frozen=True)
class DifferentiationSpec:
    forward_input_ids: tuple[int, ...]
    forward_output_id: int
    saved_value_ids: tuple[int, ...]


@dataclass(frozen=True)
class AOTContext:
    saved_values: tuple[Tensor, ...]
    output: Tensor


@dataclass(frozen=True)
class AOTForwardResult:
    output: Tensor
    context: AOTContext


@dataclass
class AOTProgram:
    forward_graph: Graph
    backward_graph: Graph
    spec: DifferentiationSpec

    def forward(self, *inputs: Tensor) -> AOTForwardResult:
        output, saved = GraphInterpreter(self.forward_graph).run_with_values(
            *inputs,
            value_ids=self.spec.saved_value_ids,
        )
        if not isinstance(output, Tensor):
            raise AOTDifferentiationError("v0.13 AOTProgram requires one Tensor output")
        return AOTForwardResult(output, AOTContext(saved_values=saved, output=output))

    def backward(self, context: AOTContext, grad_output: Tensor | None = None) -> tuple[Tensor, ...]:
        if not isinstance(context, AOTContext):
            raise TypeError("AOTProgram.backward expects the AOTContext returned by forward")
        output = context.output
        if grad_output is None:
            if output.data.size != 1:
                raise ValueError("non-scalar AOT output requires an explicit grad_output Tensor")
            grad_output = Tensor(
                np.ones_like(output.data),
                dtype=output.dtype,
                device=output.device,
                requires_grad=False,
            )
        if not isinstance(grad_output, Tensor):
            raise TypeError("grad_output must be a Tensor")
        if grad_output.shape != output.shape:
            raise ValueError(
                f"grad_output shape {grad_output.shape} does not match output shape {output.shape}"
            )
        if grad_output.device != output.device:
            raise ValueError(
                f"grad_output device {grad_output.device} does not match output device {output.device}"
            )

        # Backward-graph execution is intentionally first-order in v0.13. The
        # graph is explicit, but executing it does not construct another eager
        # autograd graph and therefore does not claim higher-order gradients.
        with no_grad():
            result = GraphInterpreter(self.backward_graph).run(
                *context.saved_values,
                grad_output,
            )
        if isinstance(result, Tensor):
            result = (result,)
        if not isinstance(result, tuple) or not all(isinstance(x, Tensor) for x in result):
            raise RuntimeError("backward Graph produced an invalid gradient structure")
        return result

    def format(self) -> str:
        return (
            "FORWARD GRAPH\n"
            + self.forward_graph.format()
            + "\n\nBACKWARD GRAPH\n"
            + self.backward_graph.format()
            + "\n\nSAVED VALUES\n"
            + repr(self.spec.saved_value_ids)
        )


def _as_ref(spec: Any, *, where: str) -> ValueRef:
    if not isinstance(spec, ValueRef):
        raise AOTDifferentiationError(f"{where} must be one Tensor ValueRef in v0.13")
    return spec


def _node_output_ref(node: Node) -> ValueRef:
    return _as_ref(node.outputs, where=f"node{node.id} output")


def _tensor_arg_ref(node: Node, index: int) -> ValueRef:
    if not isinstance(node.args, tuple) or index >= len(node.args):
        raise AOTDifferentiationError(f"node{node.id} has unsupported argument structure")
    return _as_ref(node.args[index], where=f"node{node.id} arg[{index}]")


def _grad_metadata(graph: Graph, value_id: int) -> dict[str, Any]:
    metadata = dict(graph.value(value_id).metadata)
    metadata["requires_grad"] = False
    return metadata


def _shape(graph: Graph, value_id: int) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in graph.value(value_id).metadata["shape"])
    except Exception as exc:
        raise AOTDifferentiationError(f"graph value %v{value_id} lacks shape metadata") from exc


def _requires_grad(graph: Graph, value_id: int) -> bool:
    return bool(graph.value(value_id).metadata.get("requires_grad", False))


def _saved_requirements(node: Node) -> tuple[int, ...]:
    if node.operator == "aten::mul.Tensor":
        return (_tensor_arg_ref(node, 0).value_id, _tensor_arg_ref(node, 1).value_id)
    if node.operator == "aten::matmul.default":
        return (_tensor_arg_ref(node, 0).value_id, _tensor_arg_ref(node, 1).value_id)
    if node.operator == "aten::relu.default":
        return (_tensor_arg_ref(node, 0).value_id,)
    return ()


def _reachable_and_saved(graph: Graph, output_id: int) -> tuple[set[int], tuple[int, ...]]:
    needed_grads = {output_id}
    saved: set[int] = set()
    for node in reversed(graph.nodes):
        out_id = _node_output_ref(node).value_id
        if out_id not in needed_grads:
            continue
        if node.operator not in _SUPPORTED:
            raise AOTDifferentiationError(
                f"v0.13 has no graph VJP rule for {node.operator} (node{node.id})"
            )
        saved.update(_saved_requirements(node))
        if not isinstance(node.args, tuple):
            raise AOTDifferentiationError(f"node{node.id} has unsupported argument structure")
        for item in node.args:
            if isinstance(item, ValueRef) and _requires_grad(graph, item.value_id):
                needed_grads.add(item.value_id)
    return needed_grads, tuple(sorted(saved))


class _BackwardBuilder:
    def __init__(self, forward: Graph, spec: DifferentiationSpec) -> None:
        self.forward = forward
        self.spec = spec
        self.graph = Graph()
        self.saved_refs: dict[int, ValueRef] = {}
        self.grad_map: dict[int, ValueRef] = {}

        for value_id in spec.saved_value_ids:
            value = self.graph.add_input(metadata=_grad_metadata(forward, value_id))
            self.saved_refs[value_id] = ValueRef(value.id)

        grad_out = self.graph.add_input(metadata=_grad_metadata(forward, spec.forward_output_id))
        self.grad_map[spec.forward_output_id] = ValueRef(grad_out.id)

    def _emit(self, operator: str, args: tuple[Any, ...], kwargs: dict[str, Any], *, metadata: dict[str, Any]) -> ValueRef:
        value = self.graph.add_intermediate(metadata=metadata)
        ref = ValueRef(value.id)
        self.graph.add_node(operator, args, kwargs, ref)
        return ref

    def _saved(self, value_id: int) -> ValueRef:
        try:
            return self.saved_refs[value_id]
        except KeyError as exc:
            raise AOTDifferentiationError(
                f"internal differentiation error: forward value %v{value_id} was not saved"
            ) from exc

    def _unbroadcast(self, grad: ValueRef, target_id: int) -> ValueRef:
        return self._emit(
            "aot::unbroadcast.default",
            (grad,),
            {"shape": _shape(self.forward, target_id)},
            metadata=_grad_metadata(self.forward, target_id),
        )

    def _accumulate(self, target_id: int, contribution: ValueRef) -> None:
        if not _requires_grad(self.forward, target_id):
            return
        current = self.grad_map.get(target_id)
        if current is None:
            self.grad_map[target_id] = contribution
            return
        self.grad_map[target_id] = self._emit(
            "aten::add.Tensor",
            (current, contribution),
            {},
            metadata=_grad_metadata(self.forward, target_id),
        )

    def visit(self, node: Node) -> None:
        out_id = _node_output_ref(node).value_id
        grad = self.grad_map.get(out_id)
        if grad is None:
            return

        if node.operator == "aten::add.Tensor":
            a = _tensor_arg_ref(node, 0).value_id
            b = _tensor_arg_ref(node, 1).value_id
            if _requires_grad(self.forward, a):
                self._accumulate(a, self._unbroadcast(grad, a))
            if _requires_grad(self.forward, b):
                self._accumulate(b, self._unbroadcast(grad, b))
            return

        if node.operator == "aten::mul.Tensor":
            a = _tensor_arg_ref(node, 0).value_id
            b = _tensor_arg_ref(node, 1).value_id
            if _requires_grad(self.forward, a):
                raw = self._emit(
                    "aten::mul.Tensor",
                    (grad, self._saved(b)),
                    {},
                    metadata=_grad_metadata(self.forward, out_id),
                )
                self._accumulate(a, self._unbroadcast(raw, a))
            if _requires_grad(self.forward, b):
                raw = self._emit(
                    "aten::mul.Tensor",
                    (grad, self._saved(a)),
                    {},
                    metadata=_grad_metadata(self.forward, out_id),
                )
                self._accumulate(b, self._unbroadcast(raw, b))
            return

        if node.operator == "aten::relu.default":
            a = _tensor_arg_ref(node, 0).value_id
            if _requires_grad(self.forward, a):
                contrib = self._emit(
                    "aot::relu_backward.default",
                    (grad, self._saved(a)),
                    {},
                    metadata=_grad_metadata(self.forward, a),
                )
                self._accumulate(a, contrib)
            return

        if node.operator == "aten::sum.default":
            a = _tensor_arg_ref(node, 0).value_id
            if _requires_grad(self.forward, a):
                kwargs = node.kwargs if isinstance(node.kwargs, dict) else {}
                contrib = self._emit(
                    "aot::sum_backward.default",
                    (grad,),
                    {
                        "input_shape": _shape(self.forward, a),
                        "axis": kwargs.get("axis"),
                        "keepdims": bool(kwargs.get("keepdims", False)),
                    },
                    metadata=_grad_metadata(self.forward, a),
                )
                self._accumulate(a, contrib)
            return

        if node.operator == "aten::matmul.default":
            a = _tensor_arg_ref(node, 0).value_id
            b = _tensor_arg_ref(node, 1).value_id
            if len(_shape(self.forward, a)) != 2 or len(_shape(self.forward, b)) != 2:
                raise AOTDifferentiationError("v0.13 AOT matmul VJP supports 2-D operands only")
            if _requires_grad(self.forward, a):
                bt = self._emit(
                    "aot::transpose2d.default",
                    (self._saved(b),),
                    {},
                    metadata={**_grad_metadata(self.forward, b), "shape": tuple(reversed(_shape(self.forward, b)))},
                )
                da = self._emit(
                    "aten::matmul.default",
                    (grad, bt),
                    {},
                    metadata=_grad_metadata(self.forward, a),
                )
                self._accumulate(a, da)
            if _requires_grad(self.forward, b):
                at = self._emit(
                    "aot::transpose2d.default",
                    (self._saved(a),),
                    {},
                    metadata={**_grad_metadata(self.forward, a), "shape": tuple(reversed(_shape(self.forward, a)))},
                )
                db = self._emit(
                    "aten::matmul.default",
                    (at, grad),
                    {},
                    metadata=_grad_metadata(self.forward, b),
                )
                self._accumulate(b, db)
            return

        raise AOTDifferentiationError(f"v0.13 has no graph VJP rule for {node.operator}")

    def finish(self) -> Graph:
        outputs: list[ValueRef] = []
        for value_id in self.spec.forward_input_ids:
            try:
                outputs.append(self.grad_map[value_id])
            except KeyError as exc:
                raise AOTDifferentiationError(
                    f"no gradient path from output to forward input %v{value_id}"
                ) from exc
        self.graph.outputs = tuple(outputs)
        self.graph.validate()
        return self.graph


def differentiate_graph(graph: Graph) -> AOTProgram:
    """Construct a first-order explicit backward Graph for a forward Graph."""

    if not isinstance(graph, Graph):
        raise TypeError("differentiate_graph expects a Graph")
    graph.validate()
    output_ref = _as_ref(graph.outputs, where="forward graph output")
    output_id = output_ref.value_id
    if not _requires_grad(graph, output_id):
        raise AOTDifferentiationError("forward graph output does not require grad")

    grad_inputs = tuple(value_id for value_id in graph.inputs if _requires_grad(graph, value_id))
    if not grad_inputs:
        raise AOTDifferentiationError("forward graph has no grad-requiring graph inputs")

    _, saved = _reachable_and_saved(graph, output_id)
    spec = DifferentiationSpec(
        forward_input_ids=grad_inputs,
        forward_output_id=output_id,
        saved_value_ids=saved,
    )
    builder = _BackwardBuilder(graph, spec)
    for node in reversed(graph.nodes):
        builder.visit(node)
    return AOTProgram(graph, builder.finish(), spec)


def aot_compile(fn: Callable[..., Any], *example_inputs: Tensor) -> AOTProgram:
    """Capture one forward specialization and construct its backward Graph."""

    if not example_inputs:
        raise ValueError("aot_compile requires Tensor example inputs")
    captured = capture(fn, *example_inputs)
    return differentiate_graph(captured.graph)
