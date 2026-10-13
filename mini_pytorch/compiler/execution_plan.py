"""Execution planning and conservative buffer reuse for mini-pytorch v0.15.

This module sits *after* v0.14 lowering.  It does not change graph semantics or
invent new kernels.  Instead it makes runtime lifetime/reuse decisions explicit
and lets pointwise-fusion outputs write into preallocated Storage slots when it
is provably safe to do so.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from time import perf_counter_ns
from typing import Any, Iterable

import numpy as np

from ..autograd import no_grad
from ..storage import Storage, contiguous_strides
from ..tensor import Tensor
from .analysis import analyze_graph, iter_value_ids
from .capture import is_capturing
from .lowering import ExternalOp, LoweredExecutor, LoweredProgram, LoweringError, PointwiseFusion
from .runtime_analysis import ValueLifetimeAnalysis, analyze_value_lifetimes


class ExecutionPlanningError(RuntimeError):
    """Raised when a LoweredProgram cannot form a safe execution plan."""


@dataclass(frozen=True)
class BufferSpec:
    shape: tuple[int, ...]
    dtype: str
    device: str

    @property
    def numel(self) -> int:
        return int(prod(self.shape)) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return int(self.numel * np.dtype(self.dtype).itemsize)

    def compatible_with(self, other: "BufferSpec") -> bool:
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and self.device == other.device
        )

    def format(self) -> str:
        return (
            f"BufferSpec(shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, nbytes={self.nbytes})"
        )


@dataclass(frozen=True)
class BufferInterval:
    value_id: int
    producer_node_position: int
    last_use_node_position: int
    spec: BufferSpec
    is_graph_output: bool
    is_retained: bool
    produced_by_pointwise_fusion: bool

    @property
    def eligible_for_reuse(self) -> bool:
        return (
            self.producer_node_position >= 0
            and not self.is_graph_output
            and not self.is_retained
            and self.produced_by_pointwise_fusion
        )

    def overlaps(self, other: "BufferInterval") -> bool:
        return not (
            self.last_use_node_position < other.producer_node_position
            or other.last_use_node_position < self.producer_node_position
        )

    def format(self) -> str:
        flags = []
        if self.is_graph_output:
            flags.append("output")
        if self.is_retained:
            flags.append("retained")
        if self.produced_by_pointwise_fusion:
            flags.append("pointwise")
        return (
            f"%v{self.value_id}: nodes=[{self.producer_node_position},"
            f"{self.last_use_node_position}], slot_eligible={self.eligible_for_reuse}, "
            f"flags={','.join(flags) or '-'}, {self.spec.format()}"
        )


@dataclass(frozen=True)
class BufferSlot:
    slot_id: int
    spec: BufferSpec
    value_ids: tuple[int, ...]

    @property
    def reuse_count(self) -> int:
        return max(0, len(self.value_ids) - 1)

    def format(self) -> str:
        values = ", ".join(f"%v{x}" for x in self.value_ids)
        return f"slot{self.slot_id}: {self.spec.format()}, values=[{values}]"


@dataclass(frozen=True)
class MemoryPlan:
    intervals: tuple[BufferInterval, ...]
    slots: tuple[BufferSlot, ...]
    value_to_slot: dict[int, int]
    naive_allocation_bytes: int
    planned_buffer_bytes: int
    peak_eligible_live_bytes: int

    @property
    def assigned_value_count(self) -> int:
        return len(self.value_to_slot)

    @property
    def reuse_count(self) -> int:
        return sum(slot.reuse_count for slot in self.slots)

    @property
    def bytes_avoided_by_reuse(self) -> int:
        return max(0, self.naive_allocation_bytes - self.planned_buffer_bytes)

    def slot_for(self, value_id: int) -> BufferSlot | None:
        slot_id = self.value_to_slot.get(int(value_id))
        if slot_id is None:
            return None
        return self.slots[slot_id]

    def format(self) -> str:
        lines = [
            "MemoryPlan:",
            f"  assigned_values={self.assigned_value_count}",
            f"  slots={len(self.slots)}",
            f"  reuse_count={self.reuse_count}",
            f"  naive_allocation_bytes={self.naive_allocation_bytes}",
            f"  planned_buffer_bytes={self.planned_buffer_bytes}",
            f"  bytes_avoided_by_reuse={self.bytes_avoided_by_reuse}",
            f"  peak_eligible_live_bytes={self.peak_eligible_live_bytes}",
        ]
        lines.append("  Intervals:")
        lines.extend(f"    {interval.format()}" for interval in self.intervals)
        lines.append("  Slots:")
        lines.extend(f"    {slot.format()}" for slot in self.slots)
        return "\n".join(lines)


@dataclass(frozen=True)
class ExecutionStep:
    instruction_index: int
    kind: str
    label: str
    node_ids: tuple[int, ...]
    release_after_node_ids: tuple[tuple[int, tuple[int, ...]], ...]

    def format(self) -> str:
        nodes = ",".join(f"node{x}" for x in self.node_ids)
        return (
            f"[{self.instruction_index}] {self.kind} {self.label} "
            f"nodes=[{nodes}] releases={dict(self.release_after_node_ids)}"
        )


@dataclass(frozen=True)
class ExecutionPlan:
    program: LoweredProgram
    retained_value_ids: frozenset[int]
    lifetime_analysis: ValueLifetimeAnalysis
    memory_plan: MemoryPlan
    steps: tuple[ExecutionStep, ...]
    node_position_by_id: dict[int, int]
    release_after_node: dict[int, tuple[int, ...]]

    def format(self) -> str:
        lines = [
            "ExecutionPlan:",
            f"  instructions={len(self.program.instructions)}",
            f"  graph_nodes={len(self.program.graph.nodes)}",
            "  retained=" + repr(tuple(sorted(self.retained_value_ids))),
        ]
        lines.extend(f"  {step.format()}" for step in self.steps)
        lines.append(self.memory_plan.format())
        return "\n".join(lines)


@dataclass(frozen=True)
class PlannedExecutionStats:
    slot_allocations: int
    slot_reuses: int
    planned_buffer_bytes: int
    naive_allocation_bytes: int
    bytes_avoided_by_reuse: int
    released_value_count: int
    wall_elapsed_ns: int


@dataclass(frozen=True)
class PlannedResult:
    output: Any
    stats: PlannedExecutionStats


def _metadata_spec(value) -> BufferSpec | None:
    shape = value.metadata.get("shape")
    dtype = value.metadata.get("dtype")
    device = value.metadata.get("device")
    if shape is None or dtype is None or device is None:
        return None
    try:
        normalized_shape = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in normalized_shape):
            return None
        normalized_dtype = str(np.dtype(dtype))
    except (TypeError, ValueError):
        return None
    return BufferSpec(
        shape=normalized_shape,
        dtype=normalized_dtype,
        device=str(device),
    )


def _pointwise_node_ids(program: LoweredProgram) -> frozenset[int]:
    ids: set[int] = set()
    for instruction in program.instructions:
        if isinstance(instruction, PointwiseFusion):
            ids.update(node.id for node in instruction.nodes)
    return frozenset(ids)


def _build_buffer_intervals(
    program: LoweredProgram,
    *,
    retained_value_ids: frozenset[int],
) -> tuple[BufferInterval, ...]:
    graph = program.graph
    analysis = analyze_graph(graph)
    node_position = {node.id: index for index, node in enumerate(graph.nodes)}
    pointwise_nodes = _pointwise_node_ids(program)
    graph_outputs = frozenset(iter_value_ids(graph.outputs))
    boundary = len(graph.nodes)

    intervals: list[BufferInterval] = []
    for value_id in sorted(graph.values):
        value = graph.value(value_id)
        spec = _metadata_spec(value)
        if spec is None:
            continue
        info = analysis.value(value_id)
        producer_id = info.producer_node_id
        if producer_id is None:
            producer_position = -1
            produced_by_pointwise = False
        else:
            try:
                producer_position = node_position[producer_id]
            except KeyError as exc:
                raise ExecutionPlanningError(
                    f"producer node{producer_id} for %v{value_id} is missing from Graph order"
                ) from exc
            produced_by_pointwise = producer_id in pointwise_nodes

        user_positions = []
        for user_id in info.user_node_ids:
            try:
                user_positions.append(node_position[user_id])
            except KeyError as exc:
                raise ExecutionPlanningError(
                    f"user node{user_id} for %v{value_id} is missing from Graph order"
                ) from exc
        last_use = max(user_positions) if user_positions else producer_position
        is_output = value_id in graph_outputs
        is_retained = value_id in retained_value_ids
        if is_output or is_retained:
            last_use = max(last_use, boundary)

        intervals.append(
            BufferInterval(
                value_id=value_id,
                producer_node_position=producer_position,
                last_use_node_position=last_use,
                spec=spec,
                is_graph_output=is_output,
                is_retained=is_retained,
                produced_by_pointwise_fusion=produced_by_pointwise,
            )
        )
    return tuple(intervals)


def _peak_live_bytes(intervals: Iterable[BufferInterval]) -> int:
    eligible = [interval for interval in intervals if interval.eligible_for_reuse]
    if not eligible:
        return 0
    positions = range(
        min(interval.producer_node_position for interval in eligible),
        max(interval.last_use_node_position for interval in eligible) + 1,
    )
    peak = 0
    for position in positions:
        live = sum(
            interval.spec.nbytes
            for interval in eligible
            if interval.producer_node_position <= position <= interval.last_use_node_position
        )
        peak = max(peak, live)
    return int(peak)


def plan_memory(
    program: LoweredProgram,
    *,
    retained_value_ids: Iterable[int] = (),
) -> MemoryPlan:
    """Assign reusable exact-spec Storage slots to safe pointwise temporaries.

    Reuse is deliberately conservative: only pointwise-fusion intermediates are
    eligible, output/retained Values are excluded, exact shape/dtype/device must
    match, and two assigned lifetimes may never overlap.
    """
    if not isinstance(program, LoweredProgram):
        raise TypeError("plan_memory expects a LoweredProgram")
    retained = frozenset(int(value_id) for value_id in retained_value_ids)
    missing = sorted(retained.difference(program.graph.values))
    if missing:
        text = ", ".join(f"%v{x}" for x in missing)
        raise ExecutionPlanningError(f"retained Values do not exist in Graph: {text}")

    intervals = _build_buffer_intervals(program, retained_value_ids=retained)
    candidates = sorted(
        (interval for interval in intervals if interval.eligible_for_reuse),
        key=lambda interval: (interval.producer_node_position, interval.value_id),
    )

    slot_specs: list[BufferSpec] = []
    slot_values: list[list[int]] = []
    slot_last_interval: list[BufferInterval] = []
    value_to_slot: dict[int, int] = {}

    for interval in candidates:
        chosen: int | None = None
        for slot_id, (spec, previous) in enumerate(zip(slot_specs, slot_last_interval)):
            if not spec.compatible_with(interval.spec):
                continue
            if previous.last_use_node_position < interval.producer_node_position:
                chosen = slot_id
                break
        if chosen is None:
            chosen = len(slot_specs)
            slot_specs.append(interval.spec)
            slot_values.append([])
            slot_last_interval.append(interval)
        else:
            slot_last_interval[chosen] = interval
        slot_values[chosen].append(interval.value_id)
        value_to_slot[interval.value_id] = chosen

    slots = tuple(
        BufferSlot(slot_id=index, spec=spec, value_ids=tuple(slot_values[index]))
        for index, spec in enumerate(slot_specs)
    )
    naive_bytes = sum(interval.spec.nbytes for interval in candidates)
    planned_bytes = sum(slot.spec.nbytes for slot in slots)
    return MemoryPlan(
        intervals=intervals,
        slots=slots,
        value_to_slot=value_to_slot,
        naive_allocation_bytes=int(naive_bytes),
        planned_buffer_bytes=int(planned_bytes),
        peak_eligible_live_bytes=_peak_live_bytes(intervals),
    )


def build_execution_plan(
    program: LoweredProgram,
    *,
    retained_value_ids: Iterable[int] = (),
) -> ExecutionPlan:
    if not isinstance(program, LoweredProgram):
        raise TypeError("build_execution_plan expects a LoweredProgram")
    retained = frozenset(int(value_id) for value_id in retained_value_ids)
    lifetime_analysis = analyze_value_lifetimes(
        program,
        retained_value_ids=retained,
    )
    memory_plan = plan_memory(program, retained_value_ids=retained)
    graph = program.graph
    node_position = {node.id: index for index, node in enumerate(graph.nodes)}

    release_by_position: dict[int, list[int]] = {}
    graph_outputs = frozenset(iter_value_ids(graph.outputs))
    for interval in memory_plan.intervals:
        value = graph.value(interval.value_id)
        if value.kind not in {"input", "constant", "intermediate"}:
            continue
        if interval.value_id in graph_outputs or interval.value_id in retained:
            continue
        if interval.last_use_node_position < 0 or interval.last_use_node_position >= len(graph.nodes):
            continue
        release_by_position.setdefault(interval.last_use_node_position, []).append(interval.value_id)

    release_after_node: dict[int, tuple[int, ...]] = {}
    for position, value_ids in sorted(release_by_position.items()):
        node_id = graph.nodes[position].id
        release_after_node[node_id] = tuple(sorted(set(value_ids)))

    steps: list[ExecutionStep] = []
    for index, instruction in enumerate(program.instructions):
        if isinstance(instruction, ExternalOp):
            node_ids = (instruction.node.id,)
            kind = "ExternalOp"
            label = instruction.node.operator
        elif isinstance(instruction, PointwiseFusion):
            node_ids = tuple(node.id for node in instruction.nodes)
            kind = "PointwiseFusion"
            label = f"fusion{instruction.group.id}"
        else:  # pragma: no cover
            raise ExecutionPlanningError(
                f"unsupported lowered instruction: {type(instruction).__name__}"
            )
        releases = tuple(
            (node_id, release_after_node.get(node_id, ()))
            for node_id in node_ids
            if release_after_node.get(node_id)
        )
        steps.append(
            ExecutionStep(
                instruction_index=index,
                kind=kind,
                label=label,
                node_ids=node_ids,
                release_after_node_ids=releases,
            )
        )

    return ExecutionPlan(
        program=program,
        retained_value_ids=retained,
        lifetime_analysis=lifetime_analysis,
        memory_plan=memory_plan,
        steps=tuple(steps),
        node_position_by_id=node_position,
        release_after_node=release_after_node,
    )


class PlannedExecutor(LoweredExecutor):
    """Reference runtime using an explicit ExecutionPlan.

    Only outputs of nodes inside PointwiseFusion groups may use preallocated
    slots. ExternalOps still execute through Dispatcher and own their normal
    allocation semantics. A fresh slot pool is allocated per run, so returned
    Tensors never alias a later call to the same executor.
    """

    def __init__(self, plan: ExecutionPlan) -> None:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("PlannedExecutor expects an ExecutionPlan")
        super().__init__(plan.program)
        self.plan = plan

    def _allocate_slot_storages(self) -> dict[int, Storage]:
        return {
            slot.slot_id: Storage.empty(
                slot.spec.numel,
                np.dtype(slot.spec.dtype),
                device=slot.spec.device,
            )
            for slot in self.plan.memory_plan.slots
        }

    def _tensor_for_slot(self, slot: BufferSlot, storage: Storage) -> Tensor:
        return Tensor._from_storage(
            storage,
            shape=slot.spec.shape,
            strides=contiguous_strides(slot.spec.shape),
            storage_offset=0,
            requires_grad=False,
        )

    def _release_after(self, node_id: int, env: dict[int, Tensor]) -> int:
        released = 0
        for value_id in self.plan.release_after_node.get(node_id, ()):
            if value_id in env:
                env.pop(value_id, None)
                released += 1
        return released

    def _execute_pointwise_node(
        self,
        node,
        env: dict[int, Tensor],
        slot_storages: dict[int, Storage],
    ) -> None:
        args = self._resolve(node.args, env)
        kwargs = self._resolve(node.kwargs, env)
        if kwargs:
            raise ExecutionPlanningError(
                f"planned pointwise node{node.id} unexpectedly has kwargs"
            )
        if not hasattr(node.outputs, "value_id"):
            raise ExecutionPlanningError(
                f"planned pointwise node{node.id} requires one Tensor output"
            )
        output_id = int(node.outputs.value_id)
        tensors = list(_iter_tensors(args))
        devices = {str(tensor.device) for tensor in tensors}
        if len(devices) != 1:
            raise ExecutionPlanningError(
                f"planned pointwise node{node.id} received inconsistent devices: "
                f"{sorted(devices)}"
            )
        device = next(iter(devices))

        slot = self.plan.memory_plan.slot_for(output_id)
        destination: Tensor | None = None
        if slot is not None:
            if slot.spec.device != device:
                raise ExecutionPlanningError(
                    f"slot{slot.slot_id} device {slot.spec.device} does not match runtime device {device}"
                )
            destination = self._tensor_for_slot(slot, slot_storages[slot.slot_id])

        if node.operator == "aten::add.Tensor":
            if not isinstance(args, tuple) or len(args) != 2:
                raise ExecutionPlanningError("planned add expects two Tensor arguments")
            if destination is None:
                array = args[0].data + args[1].data
            else:
                np.add(args[0].data, args[1].data, out=destination.data)
                array = None
        elif node.operator == "aten::mul.Tensor":
            if not isinstance(args, tuple) or len(args) != 2:
                raise ExecutionPlanningError("planned mul expects two Tensor arguments")
            if destination is None:
                array = args[0].data * args[1].data
            else:
                np.multiply(args[0].data, args[1].data, out=destination.data)
                array = None
        elif node.operator == "aten::relu.default":
            if not isinstance(args, tuple) or len(args) != 1:
                raise ExecutionPlanningError("planned relu expects one Tensor argument")
            if destination is None:
                array = np.maximum(args[0].data, 0)
            else:
                np.maximum(args[0].data, 0, out=destination.data)
                array = None
        else:
            raise ExecutionPlanningError(
                f"invalid operator inside planned pointwise fusion: {node.operator}"
            )

        if destination is None:
            assert array is not None
            destination = Tensor(
                np.asarray(array),
                requires_grad=False,
                dtype=np.asarray(array).dtype,
                device=device,
            )
        self._bind(node.outputs, destination, env)

    def _execute_planned(
        self,
        *inputs: Tensor,
        selected_value_ids: Iterable[int] = (),
    ) -> tuple[Any, tuple[Tensor, ...], PlannedExecutionStats]:
        if is_capturing():
            raise ExecutionPlanningError("PlannedExecutor cannot run inside graph capture")
        graph = self.program.graph
        if len(inputs) != len(graph.inputs):
            raise ValueError(
                f"planned program expects {len(graph.inputs)} inputs, got {len(inputs)}"
            )
        if not all(isinstance(value, Tensor) for value in inputs):
            raise TypeError("PlannedExecutor inputs must be Tensor objects")

        selected = tuple(int(value_id) for value_id in selected_value_ids)
        missing_selected = sorted(set(selected).difference(graph.values))
        if missing_selected:
            text = ", ".join(f"%v{x}" for x in missing_selected)
            raise ExecutionPlanningError(f"requested Values do not exist in Graph: {text}")
        not_retained = sorted(set(selected).difference(self.plan.retained_value_ids))
        if not_retained:
            text = ", ".join(f"%v{x}" for x in not_retained)
            raise ExecutionPlanningError(
                "run_with_values may only request Values retained by the ExecutionPlan: " + text
            )

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

        slot_storages = self._allocate_slot_storages()
        from ..dispatcher import dispatcher

        released_count = 0
        wall_start = perf_counter_ns()
        with no_grad():
            for instruction in self.program.instructions:
                if isinstance(instruction, ExternalOp):
                    node = instruction.node
                    args = self._resolve(node.args, env)
                    kwargs = self._resolve(node.kwargs, env)
                    result = dispatcher.dispatch(node.operator, *args, **kwargs)
                    self._bind(node.outputs, result, env)
                    released_count += self._release_after(node.id, env)
                elif isinstance(instruction, PointwiseFusion):
                    for node in instruction.nodes:
                        self._execute_pointwise_node(node, env, slot_storages)
                        released_count += self._release_after(node.id, env)
                else:  # pragma: no cover
                    raise ExecutionPlanningError(
                        f"unsupported lowered instruction: {type(instruction).__name__}"
                    )
        wall_elapsed = perf_counter_ns() - wall_start

        output = self._resolve(graph.outputs, env)
        retained_values: list[Tensor] = []
        for value_id in selected:
            try:
                retained_values.append(env[value_id])
            except KeyError as exc:
                raise ExecutionPlanningError(
                    f"retained lowered value %v{value_id} is unavailable after execution"
                ) from exc

        memory = self.plan.memory_plan
        stats = PlannedExecutionStats(
            slot_allocations=len(memory.slots),
            slot_reuses=memory.reuse_count,
            planned_buffer_bytes=memory.planned_buffer_bytes,
            naive_allocation_bytes=memory.naive_allocation_bytes,
            bytes_avoided_by_reuse=memory.bytes_avoided_by_reuse,
            released_value_count=released_count,
            wall_elapsed_ns=wall_elapsed,
        )
        return output, tuple(retained_values), stats

    def run(self, *inputs: Tensor):
        output, _, _ = self._execute_planned(*inputs)
        return output

    __call__ = run

    def run_with_stats(self, *inputs: Tensor) -> PlannedResult:
        output, _, stats = self._execute_planned(*inputs)
        return PlannedResult(output=output, stats=stats)

    def run_with_values(self, *inputs: Tensor, value_ids: Iterable[int]):
        selected = tuple(int(value_id) for value_id in value_ids)
        output, values, _ = self._execute_planned(
            *inputs,
            selected_value_ids=selected,
        )
        return output, values


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
