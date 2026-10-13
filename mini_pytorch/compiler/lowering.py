"""Educational graph lowering and pointwise fusion for mini-pytorch v0.14."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..autograd import no_grad
from ..tensor import Tensor
from .analysis import GraphAnalysis, analyze_graph, iter_value_ids
from .capture import is_capturing
from .ir import Graph, Node, TensorConstant, ValueRef, tensor_metadata


class LoweringError(RuntimeError):
    """Raised when v0.14 lowering cannot safely transform or execute a Graph."""


_POINTWISE_FUSIBLE = {
    "aten::add.Tensor",
    "aten::mul.Tensor",
    "aten::relu.default",
}
_FOLDABLE = set(_POINTWISE_FUSIBLE)


@dataclass(frozen=True)
class GraphTransform:
    graph: Graph
    value_map: dict[int, int]
    changed_node_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ConstantFoldResult:
    graph: Graph
    value_map: dict[int, int]
    folded_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class FusionGroup:
    id: int
    node_ids: tuple[int, ...]
    input_value_ids: tuple[int, ...]
    output_value_ids: tuple[int, ...]

    def format(self) -> str:
        nodes = ", ".join(f"node{x}" for x in self.node_ids)
        inputs = ", ".join(f"%v{x}" for x in self.input_value_ids)
        outputs = ", ".join(f"%v{x}" for x in self.output_value_ids)
        return (
            f"fusion{self.id}: nodes=[{nodes}], "
            f"inputs=[{inputs}], outputs=[{outputs}]"
        )


@dataclass(frozen=True)
class ExternalOp:
    node: Node

    def format(self) -> str:
        return f"ExternalOp(node{self.node.id}, {self.node.operator})"


@dataclass(frozen=True)
class PointwiseFusion:
    group: FusionGroup
    nodes: tuple[Node, ...]

    def format(self) -> str:
        operators = " -> ".join(node.operator for node in self.nodes)
        return f"PointwiseFusion(fusion{self.group.id}: {operators})"


LoweredInstruction = ExternalOp | PointwiseFusion


@dataclass(frozen=True)
class LoweredProgram:
    graph: Graph
    instructions: tuple[LoweredInstruction, ...]

    def format(self) -> str:
        lines = [
            "LoweredProgram:",
            "  inputs=" + repr(tuple(self.graph.inputs)),
            "  outputs=" + _format_spec(self.graph.outputs),
        ]
        for index, instruction in enumerate(self.instructions):
            lines.append(f"  [{index}] {instruction.format()}")
        return "\n".join(lines)


@dataclass(frozen=True)
class LoweringStats:
    original_nodes: int
    normalized_nodes: int
    live_nodes_after_dce: int
    constants_folded: int
    optimized_nodes: int
    fusion_groups: int
    fused_nodes: int
    external_nodes: int
    lowered_instructions: int

    def format(self) -> str:
        return (
            "LoweringStats("
            f"original_nodes={self.original_nodes}, "
            f"normalized_nodes={self.normalized_nodes}, "
            f"live_nodes_after_dce={self.live_nodes_after_dce}, "
            f"constants_folded={self.constants_folded}, "
            f"optimized_nodes={self.optimized_nodes}, "
            f"fusion_groups={self.fusion_groups}, "
            f"fused_nodes={self.fused_nodes}, "
            f"external_nodes={self.external_nodes}, "
            f"lowered_instructions={self.lowered_instructions})"
        )


@dataclass(frozen=True)
class LoweringResult:
    normalized_graph: Graph
    optimized_graph: Graph
    program: LoweredProgram
    fusion_groups: tuple[FusionGroup, ...]
    source_to_lowered_value: dict[int, int]
    statistics: LoweringStats

    def stats(self) -> LoweringStats:
        return self.statistics

    def mapped_value_id(self, source_value_id: int) -> int:
        try:
            return self.source_to_lowered_value[int(source_value_id)]
        except KeyError as exc:
            raise LoweringError(
                f"source graph value %v{source_value_id} was eliminated during lowering"
            ) from exc

    def explain(self) -> str:
        lines = [self.statistics.format()]
        if self.fusion_groups:
            lines.append("Fusion groups:")
            lines.extend(f"  {group.format()}" for group in self.fusion_groups)
        else:
            lines.append("Fusion groups: none")
        external = [
            instruction.node
            for instruction in self.program.instructions
            if isinstance(instruction, ExternalOp)
        ]
        if external:
            lines.append("External ops:")
            lines.extend(f"  node{node.id}: {node.operator}" for node in external)
        else:
            lines.append("External ops: none")
        return "\n".join(lines)


def _validate_graph(graph: Graph) -> None:
    if not isinstance(graph, Graph):
        raise TypeError("lowering expects a Graph")
    try:
        graph.validate()
    except Exception as exc:
        raise LoweringError(f"malformed Graph: {exc}") from exc


def _clone_constant(payload: TensorConstant) -> TensorConstant:
    return TensorConstant(
        data=np.array(payload.data, copy=True),
        requires_grad=bool(payload.requires_grad),
        device=str(payload.device),
    )


def _map_spec(spec: Any, value_map: dict[int, int]) -> Any:
    if isinstance(spec, ValueRef):
        try:
            return ValueRef(value_map[spec.value_id])
        except KeyError as exc:
            raise LoweringError(
                f"missing mapped dependency for graph value %v{spec.value_id}"
            ) from exc
    if isinstance(spec, tuple):
        return tuple(_map_spec(x, value_map) for x in spec)
    if isinstance(spec, list):
        return [_map_spec(x, value_map) for x in spec]
    if isinstance(spec, dict):
        return {
            _map_spec(key, value_map): _map_spec(value, value_map)
            for key, value in spec.items()
        }
    return spec


def _create_output_values(dst: Graph, src: Graph, spec: Any, value_map: dict[int, int]) -> Any:
    if isinstance(spec, ValueRef):
        old_id = spec.value_id
        if old_id in value_map:
            return ValueRef(value_map[old_id])
        source = src.value(old_id)
        if source.kind != "intermediate":
            raise LoweringError(
                f"node output %v{old_id} is {source.kind}, expected intermediate"
            )
        created = dst.add_intermediate(metadata=dict(source.metadata))
        value_map[old_id] = created.id
        return ValueRef(created.id)
    if isinstance(spec, tuple):
        return tuple(_create_output_values(dst, src, x, value_map) for x in spec)
    if isinstance(spec, list):
        return [_create_output_values(dst, src, x, value_map) for x in spec]
    raise LoweringError(f"unsupported node output structure: {spec!r}")


def _clone_selected(
    graph: Graph,
    *,
    keep_node_ids: set[int] | None = None,
    keep_constant_ids: set[int] | None = None,
) -> GraphTransform:
    _validate_graph(graph)
    dst = Graph()
    value_map: dict[int, int] = {}

    for old_id in graph.inputs:
        source = graph.value(old_id)
        value_map[old_id] = dst.add_input(metadata=dict(source.metadata)).id

    for old_id in sorted(graph.values):
        source = graph.values[old_id]
        if source.kind != "constant":
            continue
        if keep_constant_ids is not None and old_id not in keep_constant_ids:
            continue
        assert source.constant is not None
        value_map[old_id] = dst.add_constant(
            _clone_constant(source.constant),
            metadata=dict(source.metadata),
        ).id

    changed: list[int] = []
    for node in graph.nodes:
        if keep_node_ids is not None and node.id not in keep_node_ids:
            changed.append(node.id)
            continue
        args = _map_spec(node.args, value_map)
        kwargs = _map_spec(node.kwargs, value_map)
        outputs = _create_output_values(dst, graph, node.outputs, value_map)
        dst.add_node(node.operator, args, kwargs, outputs)

    dst.outputs = _map_spec(graph.outputs, value_map)
    _validate_graph(dst)
    return GraphTransform(dst, value_map, tuple(changed))


def _normalize_with_map(graph: Graph) -> GraphTransform:
    return _clone_selected(graph)


def normalize_graph(graph: Graph) -> Graph:
    """Return a deterministic deep Graph clone without mutating the source."""
    return _normalize_with_map(graph).graph


def _dce_with_map(graph: Graph) -> GraphTransform:
    analysis = analyze_graph(graph)
    live_nodes = set(analysis.live_node_ids)
    live_constants = {
        value_id
        for value_id in analysis.live_value_ids
        if graph.value(value_id).kind == "constant"
    }
    return _clone_selected(
        graph,
        keep_node_ids=live_nodes,
        keep_constant_ids=live_constants,
    )


def eliminate_dead_code(graph: Graph) -> Graph:
    """Remove pure nodes/constant Values unreachable from graph outputs."""
    return _dce_with_map(graph).graph


def _resolve_constant_spec(spec: Any, graph: Graph) -> Any:
    if isinstance(spec, ValueRef):
        value = graph.value(spec.value_id)
        if value.kind != "constant" or value.constant is None:
            raise LoweringError("constant-fold evaluation received a runtime Value")
        payload = value.constant
        return Tensor(
            np.array(payload.data, copy=True),
            requires_grad=False,
            dtype=payload.data.dtype,
            device=payload.device,
        )
    if isinstance(spec, tuple):
        return tuple(_resolve_constant_spec(x, graph) for x in spec)
    if isinstance(spec, list):
        return [_resolve_constant_spec(x, graph) for x in spec]
    if isinstance(spec, dict):
        return {
            _resolve_constant_spec(key, graph): _resolve_constant_spec(value, graph)
            for key, value in spec.items()
        }
    return spec


def _can_fold(
    operator: str,
    args: Any,
    kwargs: Any,
    graph: Graph,
    *,
    output_metadata: dict[str, Any],
) -> bool:
    if operator not in _FOLDABLE:
        return False
    refs = list(iter_value_ids(args)) + list(iter_value_ids(kwargs))
    if not refs:
        return False
    devices: set[str] = set()
    for value_id in refs:
        value = graph.value(value_id)
        if value.kind != "constant" or value.constant is None:
            return False
        if value.constant.requires_grad or value.metadata.get("requires_grad", False):
            return False
        devices.add(str(value.constant.device))
    if len(devices) != 1:
        return False
    if output_metadata.get("requires_grad", False):
        return False
    return True


def fold_constants(graph: Graph) -> ConstantFoldResult:
    """Fold safe pure add/mul/relu nodes with constant-only Tensor inputs."""
    _validate_graph(graph)
    from ..dispatcher import dispatcher

    dst = Graph()
    value_map: dict[int, int] = {}

    for old_id in graph.inputs:
        source = graph.value(old_id)
        value_map[old_id] = dst.add_input(metadata=dict(source.metadata)).id

    for old_id in sorted(graph.values):
        source = graph.values[old_id]
        if source.kind != "constant":
            continue
        assert source.constant is not None
        value_map[old_id] = dst.add_constant(
            _clone_constant(source.constant),
            metadata=dict(source.metadata),
        ).id

    folded: list[int] = []
    for node in graph.nodes:
        mapped_args = _map_spec(node.args, value_map)
        mapped_kwargs = _map_spec(node.kwargs, value_map)
        if not isinstance(node.outputs, ValueRef):
            output_metadata: dict[str, Any] = {}
        else:
            output_metadata = dict(graph.value(node.outputs.value_id).metadata)

        if _can_fold(
            node.operator,
            mapped_args,
            mapped_kwargs,
            dst,
            output_metadata=output_metadata,
        ):
            with no_grad():
                args = _resolve_constant_spec(mapped_args, dst)
                kwargs = _resolve_constant_spec(mapped_kwargs, dst)
                result = dispatcher.dispatch(node.operator, *args, **kwargs)
            if not isinstance(result, Tensor):
                raise LoweringError(
                    f"constant-folded {node.operator} returned {type(result).__name__}"
                )
            if not isinstance(node.outputs, ValueRef):
                raise LoweringError("v0.14 constant folding expects one Tensor output")
            created = dst.add_constant(
                TensorConstant(
                    data=np.array(result.data, copy=True),
                    requires_grad=False,
                    device=str(result.device),
                ),
                metadata=tensor_metadata(result),
            )
            value_map[node.outputs.value_id] = created.id
            folded.append(node.id)
            continue

        outputs = _create_output_values(dst, graph, node.outputs, value_map)
        dst.add_node(node.operator, mapped_args, mapped_kwargs, outputs)

    dst.outputs = _map_spec(graph.outputs, value_map)
    _validate_graph(dst)
    return ConstantFoldResult(dst, value_map, tuple(folded))


def _single_output_id(node: Node) -> int | None:
    if isinstance(node.outputs, ValueRef):
        return int(node.outputs.value_id)
    return None


def _ordered_external_inputs(
    nodes: Iterable[Node],
    group_node_ids: set[int],
    analysis: GraphAnalysis,
) -> tuple[int, ...]:
    result: list[int] = []
    seen: set[int] = set()
    for node in nodes:
        for value_id in list(iter_value_ids(node.args)) + list(iter_value_ids(node.kwargs)):
            producer = analysis.value(value_id).producer_node_id
            if producer in group_node_ids:
                continue
            if value_id not in seen:
                seen.add(value_id)
                result.append(value_id)
    return tuple(result)


def _group_outputs(
    nodes: Iterable[Node],
    group_node_ids: set[int],
    analysis: GraphAnalysis,
) -> tuple[int, ...]:
    result: list[int] = []
    seen: set[int] = set()
    for node in nodes:
        for value_id in iter_value_ids(node.outputs):
            info = analysis.value(value_id)
            escapes = info.is_graph_output or any(
                user not in group_node_ids for user in info.user_node_ids
            )
            if escapes and value_id not in seen:
                seen.add(value_id)
                result.append(value_id)
    return tuple(result)


def plan_fusions(graph: Graph, analysis: GraphAnalysis | None = None) -> tuple[FusionGroup, ...]:
    _validate_graph(graph)
    analysis = analysis or analyze_graph(graph)
    nodes = graph.nodes
    groups: list[FusionGroup] = []
    index = 0

    while index < len(nodes):
        first = nodes[index]
        if first.operator not in _POINTWISE_FUSIBLE or _single_output_id(first) is None:
            index += 1
            continue

        chain = [first]
        cursor = index
        while cursor + 1 < len(nodes):
            current = chain[-1]
            next_node = nodes[cursor + 1]
            current_output = _single_output_id(current)
            if current_output is None:
                break
            info = analysis.value(current_output)
            if info.use_count != 1 or info.is_graph_output:
                break
            if next_node.operator not in _POINTWISE_FUSIBLE:
                break
            if _single_output_id(next_node) is None:
                break
            next_refs = set(iter_value_ids(next_node.args)) | set(iter_value_ids(next_node.kwargs))
            if current_output not in next_refs:
                break
            chain.append(next_node)
            cursor += 1

        if len(chain) >= 2:
            node_ids = {node.id for node in chain}
            outputs = _group_outputs(chain, node_ids, analysis)
            if not outputs:
                raise LoweringError("fusion group has no externally visible output")
            groups.append(
                FusionGroup(
                    id=len(groups),
                    node_ids=tuple(node.id for node in chain),
                    input_value_ids=_ordered_external_inputs(chain, node_ids, analysis),
                    output_value_ids=outputs,
                )
            )
            index = cursor + 1
        else:
            index += 1

    return tuple(groups)


def _build_program(graph: Graph, groups: tuple[FusionGroup, ...]) -> LoweredProgram:
    group_by_first = {group.node_ids[0]: group for group in groups}
    grouped_node_ids = {node_id for group in groups for node_id in group.node_ids}
    node_by_id = {node.id: node for node in graph.nodes}

    instructions: list[LoweredInstruction] = []
    for node in graph.nodes:
        group = group_by_first.get(node.id)
        if group is not None:
            instructions.append(
                PointwiseFusion(
                    group=group,
                    nodes=tuple(node_by_id[node_id] for node_id in group.node_ids),
                )
            )
            continue
        if node.id in grouped_node_ids:
            continue
        instructions.append(ExternalOp(node))
    return LoweredProgram(graph, tuple(instructions))


def _compose_maps(*maps: dict[int, int]) -> dict[int, int]:
    if not maps:
        return {}
    current = dict(maps[0])
    for next_map in maps[1:]:
        composed: dict[int, int] = {}
        for source, intermediate in current.items():
            if intermediate in next_map:
                composed[source] = next_map[intermediate]
        current = composed
    return current


def lower_graph(graph: Graph) -> LoweringResult:
    """Run the v0.14 lowering pipeline and return an inspectable execution plan."""
    _validate_graph(graph)

    normalized_result = _normalize_with_map(graph)
    normalized = normalized_result.graph

    dce_result = _dce_with_map(normalized)
    after_dce = dce_result.graph

    fold_result = fold_constants(after_dce)
    after_fold = fold_result.graph

    final_dce_result = _dce_with_map(after_fold)
    optimized = final_dce_result.graph

    analysis = analyze_graph(optimized)
    groups = plan_fusions(optimized, analysis)
    program = _build_program(optimized, groups)

    fused_node_ids = {node_id for group in groups for node_id in group.node_ids}
    external_nodes = len(optimized.nodes) - len(fused_node_ids)
    stats = LoweringStats(
        original_nodes=len(graph.nodes),
        normalized_nodes=len(normalized.nodes),
        live_nodes_after_dce=len(after_dce.nodes),
        constants_folded=len(fold_result.folded_node_ids),
        optimized_nodes=len(optimized.nodes),
        fusion_groups=len(groups),
        fused_nodes=len(fused_node_ids),
        external_nodes=external_nodes,
        lowered_instructions=len(program.instructions),
    )
    value_map = _compose_maps(
        normalized_result.value_map,
        dce_result.value_map,
        fold_result.value_map,
        final_dce_result.value_map,
    )
    return LoweringResult(
        normalized_graph=normalized,
        optimized_graph=optimized,
        program=program,
        fusion_groups=groups,
        source_to_lowered_value=value_map,
        statistics=stats,
    )


class LoweredExecutor:
    """Reference executor for LoweredProgram.

    v0.14 execution is first-order and numerical. Fused groups execute without
    Dispatcher calls using NumPy reference expressions; ExternalOps continue to
    use Dispatcher. The full run is under no_grad so lowering never creates a
    partial eager Autograd graph.
    """

    def __init__(self, program: LoweredProgram) -> None:
        if not isinstance(program, LoweredProgram):
            raise TypeError("LoweredExecutor expects a LoweredProgram")
        _validate_graph(program.graph)
        self.program = program

    def _execute(self, *inputs: Tensor) -> tuple[Any, dict[int, Tensor]]:
        if is_capturing():
            raise LoweringError("LoweredExecutor cannot run inside graph capture")
        graph = self.program.graph
        if len(inputs) != len(graph.inputs):
            raise ValueError(
                f"lowered program expects {len(graph.inputs)} inputs, got {len(inputs)}"
            )
        if not all(isinstance(value, Tensor) for value in inputs):
            raise TypeError("LoweredExecutor inputs must be Tensor objects")

        env: dict[int, Tensor] = {}
        for value_id, tensor in zip(graph.inputs, inputs):
            env[value_id] = tensor

        for value_id, value in graph.values.items():
            if value.kind != "constant":
                continue
            payload = value.constant
            assert payload is not None
            env[value_id] = Tensor(
                np.array(payload.data, copy=True),
                requires_grad=False,
                dtype=payload.data.dtype,
                device=payload.device,
            )

        from ..dispatcher import dispatcher

        with no_grad():
            for instruction in self.program.instructions:
                if isinstance(instruction, ExternalOp):
                    node = instruction.node
                    args = self._resolve(node.args, env)
                    kwargs = self._resolve(node.kwargs, env)
                    result = dispatcher.dispatch(node.operator, *args, **kwargs)
                    self._bind(node.outputs, result, env)
                else:
                    self._run_fusion(instruction, env)

        return self._resolve(graph.outputs, env), env

    def run(self, *inputs: Tensor):
        result, _ = self._execute(*inputs)
        return result

    __call__ = run

    def run_with_values(self, *inputs: Tensor, value_ids: Iterable[int]):
        output, env = self._execute(*inputs)
        selected = []
        for value_id in value_ids:
            value_id = int(value_id)
            try:
                selected.append(env[value_id])
            except KeyError as exc:
                raise LoweringError(
                    f"requested lowered value %v{value_id} is unavailable"
                ) from exc
        return output, tuple(selected)

    def _run_fusion(self, instruction: PointwiseFusion, env: dict[int, Tensor]) -> None:
        for node in instruction.nodes:
            args = self._resolve(node.args, env)
            kwargs = self._resolve(node.kwargs, env)
            if kwargs:
                raise LoweringError(
                    f"pointwise fusion node{node.id} unexpectedly has kwargs"
                )
            tensors = list(_iter_tensors(args))
            devices = {str(tensor.device) for tensor in tensors}
            if len(devices) != 1:
                raise LoweringError(
                    f"pointwise fusion node{node.id} received inconsistent devices: "
                    f"{sorted(devices)}"
                )
            device = next(iter(devices))

            if node.operator == "aten::add.Tensor":
                if not isinstance(args, tuple) or len(args) != 2:
                    raise LoweringError("fused add expects two Tensor arguments")
                array = args[0].data + args[1].data
            elif node.operator == "aten::mul.Tensor":
                if not isinstance(args, tuple) or len(args) != 2:
                    raise LoweringError("fused mul expects two Tensor arguments")
                array = args[0].data * args[1].data
            elif node.operator == "aten::relu.default":
                if not isinstance(args, tuple) or len(args) != 1:
                    raise LoweringError("fused relu expects one Tensor argument")
                array = np.maximum(args[0].data, 0)
            else:
                raise LoweringError(
                    f"invalid operator inside pointwise fusion: {node.operator}"
                )

            result = Tensor(
                np.asarray(array),
                requires_grad=False,
                dtype=np.asarray(array).dtype,
                device=device,
            )
            self._bind(node.outputs, result, env)

    def _resolve(self, spec: Any, env: dict[int, Tensor]) -> Any:
        if isinstance(spec, ValueRef):
            try:
                return env[spec.value_id]
            except KeyError as exc:
                raise LoweringError(
                    f"lowered value %v{spec.value_id} is not available"
                ) from exc
        if isinstance(spec, tuple):
            return tuple(self._resolve(x, env) for x in spec)
        if isinstance(spec, list):
            return [self._resolve(x, env) for x in spec]
        if isinstance(spec, dict):
            return {
                self._resolve(key, env): self._resolve(value, env)
                for key, value in spec.items()
            }
        return spec

    def _bind(self, spec: Any, result: Any, env: dict[int, Tensor]) -> None:
        if isinstance(spec, ValueRef):
            if not isinstance(result, Tensor):
                raise LoweringError(
                    f"lowered node output expected Tensor, got {type(result).__name__}"
                )
            env[spec.value_id] = result
            return
        if isinstance(spec, tuple):
            if not isinstance(result, tuple) or len(spec) != len(result):
                raise LoweringError("lowered output structure mismatch (tuple)")
            for child, value in zip(spec, result):
                self._bind(child, value, env)
            return
        if isinstance(spec, list):
            if not isinstance(result, list) or len(spec) != len(result):
                raise LoweringError("lowered output structure mismatch (list)")
            for child, value in zip(spec, result):
                self._bind(child, value, env)
            return
        raise LoweringError(f"unsupported lowered output binding spec: {spec!r}")


def _iter_tensors(spec: Any):
    if isinstance(spec, Tensor):
        yield spec
        return
    if isinstance(spec, (tuple, list)):
        for item in spec:
            yield from _iter_tensors(item)
        return
    if isinstance(spec, dict):
        for key, value in spec.items():
            yield from _iter_tensors(key)
            yield from _iter_tensors(value)


def _format_spec(spec: Any) -> str:
    if isinstance(spec, ValueRef):
        return f"%v{spec.value_id}"
    if isinstance(spec, tuple):
        return "(" + ", ".join(_format_spec(x) for x in spec) + ")"
    if isinstance(spec, list):
        return "[" + ", ".join(_format_spec(x) for x in spec) + "]"
    if isinstance(spec, dict):
        return "{" + ", ".join(
            f"{_format_spec(key)}: {_format_spec(value)}"
            for key, value in sorted(spec.items(), key=lambda item: repr(item[0]))
        ) + "}"
    return repr(spec)
