"""Lowered-program runtime analysis for mini-pytorch v0.15."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable

import numpy as np

from .analysis import iter_value_ids
from .lowering import ExternalOp, LoweredProgram, PointwiseFusion


class RuntimeAnalysisError(RuntimeError):
    """Raised when a LoweredProgram cannot form a valid runtime lifetime model."""


def _unique(ids: Iterable[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value_id in ids:
        value_id = int(value_id)
        if value_id not in seen:
            seen.add(value_id)
            ordered.append(value_id)
    return tuple(ordered)


def instruction_input_value_ids(instruction: ExternalOp | PointwiseFusion) -> tuple[int, ...]:
    """Values entering one lowered instruction from outside that instruction."""
    if isinstance(instruction, ExternalOp):
        node = instruction.node
        return _unique(
            list(iter_value_ids(node.args)) + list(iter_value_ids(node.kwargs))
        )
    if isinstance(instruction, PointwiseFusion):
        return tuple(instruction.group.input_value_ids)
    raise TypeError(f"unsupported lowered instruction: {type(instruction).__name__}")


def instruction_output_value_ids(instruction: ExternalOp | PointwiseFusion) -> tuple[int, ...]:
    """Values escaping one lowered instruction."""
    if isinstance(instruction, ExternalOp):
        return _unique(iter_value_ids(instruction.node.outputs))
    if isinstance(instruction, PointwiseFusion):
        return tuple(instruction.group.output_value_ids)
    raise TypeError(f"unsupported lowered instruction: {type(instruction).__name__}")


def instruction_produced_value_ids(instruction: ExternalOp | PointwiseFusion) -> tuple[int, ...]:
    """All Values materialized by an instruction, including fused internals."""
    if isinstance(instruction, ExternalOp):
        return _unique(iter_value_ids(instruction.node.outputs))
    if isinstance(instruction, PointwiseFusion):
        ids: list[int] = []
        for node in instruction.nodes:
            ids.extend(iter_value_ids(node.outputs))
        return _unique(ids)
    raise TypeError(f"unsupported lowered instruction: {type(instruction).__name__}")


def instruction_referenced_value_ids(instruction: ExternalOp | PointwiseFusion) -> tuple[int, ...]:
    """All Value dependencies read inside an instruction, including fused internals."""
    if isinstance(instruction, ExternalOp):
        node = instruction.node
        return _unique(
            list(iter_value_ids(node.args)) + list(iter_value_ids(node.kwargs))
        )
    if isinstance(instruction, PointwiseFusion):
        ids: list[int] = []
        for node in instruction.nodes:
            ids.extend(iter_value_ids(node.args))
            ids.extend(iter_value_ids(node.kwargs))
        return _unique(ids)
    raise TypeError(f"unsupported lowered instruction: {type(instruction).__name__}")


@dataclass(frozen=True)
class ValueLifetime:
    value_id: int
    kind: str
    birth_instruction: int
    last_use_instruction: int
    is_input: bool
    is_constant: bool
    is_output: bool
    is_retained: bool
    shape: tuple[int, ...] | None
    dtype: str | None
    device: str | None

    @property
    def numel(self) -> int | None:
        if self.shape is None:
            return None
        return int(prod(self.shape)) if self.shape else 1

    @property
    def estimated_nbytes(self) -> int | None:
        if self.numel is None or self.dtype is None:
            return None
        try:
            return int(self.numel * np.dtype(self.dtype).itemsize)
        except TypeError:
            return None

    def format(self) -> str:
        flags = []
        if self.is_input:
            flags.append("input")
        if self.is_constant:
            flags.append("constant")
        if self.is_output:
            flags.append("output")
        if self.is_retained:
            flags.append("retained")
        flag_text = ",".join(flags) or "internal"
        return (
            f"%v{self.value_id}: kind={self.kind}, "
            f"life=[{self.birth_instruction},{self.last_use_instruction}], "
            f"flags={flag_text}, shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}"
        )


@dataclass(frozen=True)
class ValueLifetimeAnalysis:
    instruction_count: int
    retained_value_ids: frozenset[int]
    lifetimes: tuple[ValueLifetime, ...]

    def value(self, value_id: int) -> ValueLifetime:
        value_id = int(value_id)
        for lifetime in self.lifetimes:
            if lifetime.value_id == value_id:
                return lifetime
        raise KeyError(f"no lifetime for lowered value %v{value_id}")

    def format(self) -> str:
        lines = [
            "ValueLifetimeAnalysis:",
            f"  instruction_count={self.instruction_count}",
            "  retained=" + repr(tuple(sorted(self.retained_value_ids))),
        ]
        lines.extend(f"  {lifetime.format()}" for lifetime in self.lifetimes)
        return "\n".join(lines)


def analyze_value_lifetimes(
    program: LoweredProgram,
    *,
    retained_value_ids: Iterable[int] = (),
) -> ValueLifetimeAnalysis:
    """Compute coarse instruction-level lifetimes for all lowered Graph Values.

    Fused node internals have the same instruction-level birth/use index because
    one `PointwiseFusion` is one lowered instruction. They are still represented
    because the v0.14 reference executor materializes them and AOT may request a
    saved internal Value after forward execution.
    """

    if not isinstance(program, LoweredProgram):
        raise TypeError("analyze_value_lifetimes expects a LoweredProgram")
    try:
        program.graph.validate()
    except Exception as exc:
        raise RuntimeAnalysisError(f"malformed lowered Graph: {exc}") from exc

    graph = program.graph
    retained = frozenset(int(value_id) for value_id in retained_value_ids)
    missing_retained = sorted(retained.difference(graph.values))
    if missing_retained:
        text = ", ".join(f"%v{x}" for x in missing_retained)
        raise RuntimeAnalysisError(f"retained Values do not exist in Graph: {text}")

    producer_instruction: dict[int, int] = {}
    for value_id in graph.inputs:
        producer_instruction[value_id] = -1
    for value_id, value in graph.values.items():
        if value.kind == "constant":
            producer_instruction[value_id] = -1

    for index, instruction in enumerate(program.instructions):
        for value_id in instruction_produced_value_ids(instruction):
            if value_id in producer_instruction:
                raise RuntimeAnalysisError(
                    f"lowered value %v{value_id} has multiple runtime producers"
                )
            producer_instruction[value_id] = index

    for value_id, value in graph.values.items():
        if value_id not in producer_instruction:
            raise RuntimeAnalysisError(
                f"lowered value %v{value_id} ({value.kind}) has no runtime producer"
            )

    uses: dict[int, list[int]] = {value_id: [] for value_id in graph.values}
    for index, instruction in enumerate(program.instructions):
        for value_id in instruction_referenced_value_ids(instruction):
            if value_id not in graph.values:
                raise RuntimeAnalysisError(
                    f"instruction {index} references missing value %v{value_id}"
                )
            producer = producer_instruction[value_id]
            if producer > index:
                raise RuntimeAnalysisError(
                    f"instruction {index} uses %v{value_id} before producer instruction {producer}"
                )
            uses[value_id].append(index)

    output_ids = frozenset(iter_value_ids(graph.outputs))
    boundary = len(program.instructions)
    lifetimes: list[ValueLifetime] = []

    for value_id in sorted(graph.values):
        value = graph.values[value_id]
        birth = producer_instruction[value_id]
        value_uses = uses[value_id]
        last_use = max(value_uses) if value_uses else birth
        is_output = value_id in output_ids
        is_retained = value_id in retained
        if is_output or is_retained:
            last_use = max(last_use, boundary)
        if last_use < birth:
            raise RuntimeAnalysisError(
                f"invalid lifetime for %v{value_id}: birth={birth}, last_use={last_use}"
            )

        shape_value = value.metadata.get("shape")
        shape = None if shape_value is None else tuple(int(x) for x in shape_value)
        dtype_value = value.metadata.get("dtype")
        device_value = value.metadata.get("device")
        lifetimes.append(
            ValueLifetime(
                value_id=value_id,
                kind=value.kind,
                birth_instruction=birth,
                last_use_instruction=last_use,
                is_input=value.kind == "input",
                is_constant=value.kind == "constant",
                is_output=is_output,
                is_retained=is_retained,
                shape=shape,
                dtype=None if dtype_value is None else str(dtype_value),
                device=None if device_value is None else str(device_value),
            )
        )

    return ValueLifetimeAnalysis(
        instruction_count=boundary,
        retained_value_ids=retained,
        lifetimes=tuple(lifetimes),
    )
