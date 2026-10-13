"""Optional per-instruction profiling for LoweredProgram execution."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

import numpy as np

from ..autograd import no_grad
from ..tensor import Tensor
from .capture import is_capturing
from .lowering import (
    ExternalOp,
    LoweredExecutor,
    LoweredProgram,
    LoweringError,
    PointwiseFusion,
)
from .runtime_analysis import (
    instruction_input_value_ids,
    instruction_output_value_ids,
)


@dataclass(frozen=True)
class ExecutionEvent:
    instruction_index: int
    instruction_kind: str
    label: str
    input_value_ids: tuple[int, ...]
    output_value_ids: tuple[int, ...]
    elapsed_ns: int

    def __post_init__(self) -> None:
        if self.instruction_index < 0:
            raise ValueError("instruction_index must be >= 0")
        if self.elapsed_ns < 0:
            raise ValueError("elapsed_ns must be non-negative")

    def format(self) -> str:
        inputs = ", ".join(f"%v{x}" for x in self.input_value_ids) or "-"
        outputs = ", ".join(f"%v{x}" for x in self.output_value_ids) or "-"
        return (
            f"[{self.instruction_index}] {self.instruction_kind} {self.label}: "
            f"inputs=[{inputs}], outputs=[{outputs}], elapsed_ns={self.elapsed_ns}"
        )


@dataclass(frozen=True)
class ExecutionProfile:
    events: tuple[ExecutionEvent, ...]
    wall_elapsed_ns: int

    def __post_init__(self) -> None:
        if self.wall_elapsed_ns < 0:
            raise ValueError("wall_elapsed_ns must be non-negative")

    @property
    def instruction_elapsed_ns(self) -> int:
        return int(sum(event.elapsed_ns for event in self.events))

    def by_kind(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for event in self.events:
            totals[event.instruction_kind] = totals.get(event.instruction_kind, 0) + event.elapsed_ns
        return totals

    def format(self) -> str:
        lines = [
            "ExecutionProfile:",
            f"  wall_elapsed_ns={self.wall_elapsed_ns}",
            f"  instruction_elapsed_ns={self.instruction_elapsed_ns}",
        ]
        lines.extend(f"  {event.format()}" for event in self.events)
        return "\n".join(lines)


@dataclass(frozen=True)
class ProfiledResult:
    output: Any
    profile: ExecutionProfile


def _instruction_label(instruction: ExternalOp | PointwiseFusion) -> tuple[str, str]:
    if isinstance(instruction, ExternalOp):
        return "ExternalOp", instruction.node.operator
    if isinstance(instruction, PointwiseFusion):
        ops = "->".join(node.operator for node in instruction.nodes)
        return "PointwiseFusion", f"fusion{instruction.group.id}[{ops}]"
    raise TypeError(f"unsupported lowered instruction: {type(instruction).__name__}")


class _ProfilingLoweredExecutor(LoweredExecutor):
    """Instrumented control loop reusing LoweredExecutor numerical helpers."""

    def run_profiled(self, *inputs: Tensor) -> ProfiledResult:
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

        events: list[ExecutionEvent] = []
        wall_start = perf_counter_ns()
        with no_grad():
            for index, instruction in enumerate(self.program.instructions):
                start = perf_counter_ns()
                if isinstance(instruction, ExternalOp):
                    node = instruction.node
                    args = self._resolve(node.args, env)
                    kwargs = self._resolve(node.kwargs, env)
                    result = dispatcher.dispatch(node.operator, *args, **kwargs)
                    self._bind(node.outputs, result, env)
                elif isinstance(instruction, PointwiseFusion):
                    self._run_fusion(instruction, env)
                else:  # pragma: no cover - LoweredProgram currently has a closed union
                    raise LoweringError(
                        f"unsupported lowered instruction: {type(instruction).__name__}"
                    )
                elapsed = perf_counter_ns() - start
                kind, label = _instruction_label(instruction)
                events.append(
                    ExecutionEvent(
                        instruction_index=index,
                        instruction_kind=kind,
                        label=label,
                        input_value_ids=instruction_input_value_ids(instruction),
                        output_value_ids=instruction_output_value_ids(instruction),
                        elapsed_ns=elapsed,
                    )
                )
        wall_elapsed = perf_counter_ns() - wall_start

        output = self._resolve(graph.outputs, env)
        return ProfiledResult(
            output=output,
            profile=ExecutionProfile(
                events=tuple(events),
                wall_elapsed_ns=wall_elapsed,
            ),
        )


def profile_lowered(program: LoweredProgram, *inputs: Tensor) -> ProfiledResult:
    """Execute a LoweredProgram while recording one timing event per instruction."""
    if not isinstance(program, LoweredProgram):
        raise TypeError("profile_lowered expects a LoweredProgram")
    return _ProfilingLoweredExecutor(program).run_profiled(*inputs)
