"""Inspectable specialization guards for mini-pytorch v0.12."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .signature import InputSignature


@dataclass(frozen=True)
class GuardResult:
    label: str
    passed: bool
    expected: Any
    actual: Any
    reason: str

    def format(self) -> str:
        state = "PASS" if self.passed else "FAIL"
        return f"{state} {self.label}: expected {self.expected!r}, actual {self.actual!r}"


class Guard(Protocol):
    def check(self, signature: InputSignature) -> GuardResult: ...
    def format(self) -> str: ...


@dataclass(frozen=True)
class InputCountGuard:
    expected: int

    def check(self, signature: InputSignature) -> GuardResult:
        actual = len(signature.tensors)
        passed = actual == self.expected
        return GuardResult(
            "input_count",
            passed,
            self.expected,
            actual,
            "input count matches" if passed else "different number of graph inputs",
        )

    def format(self) -> str:
        return f"input_count == {self.expected}"


@dataclass(frozen=True)
class _TensorFieldGuard:
    index: int
    expected: Any
    field: str

    def check(self, signature: InputSignature) -> GuardResult:
        if self.index >= len(signature.tensors):
            return GuardResult(
                f"input[{self.index}].{self.field}",
                False,
                self.expected,
                "<missing>",
                "graph input is missing",
            )
        actual = getattr(signature.tensors[self.index], self.field)
        passed = actual == self.expected
        return GuardResult(
            f"input[{self.index}].{self.field}",
            passed,
            self.expected,
            actual,
            "guard passed" if passed else f"{self.field} specialization mismatch",
        )

    def format(self) -> str:
        return f"input[{self.index}].{self.field} == {self.expected!r}"


class TensorShapeGuard(_TensorFieldGuard):
    def __init__(self, index: int, expected: tuple[int, ...]) -> None:
        super().__init__(index, tuple(expected), "shape")


class TensorDTypeGuard(_TensorFieldGuard):
    def __init__(self, index: int, expected: str) -> None:
        super().__init__(index, str(expected), "dtype")


class TensorDeviceGuard(_TensorFieldGuard):
    def __init__(self, index: int, expected: str) -> None:
        super().__init__(index, str(expected), "device")


class RequiresGradGuard(_TensorFieldGuard):
    def __init__(self, index: int, expected: bool) -> None:
        super().__init__(index, bool(expected), "requires_grad")


@dataclass(frozen=True)
class GradModeGuard:
    expected: bool

    def check(self, signature: InputSignature) -> GuardResult:
        actual = bool(signature.grad_enabled)
        passed = actual == self.expected
        return GuardResult(
            "grad_enabled",
            passed,
            self.expected,
            actual,
            "grad mode matches" if passed else "grad-mode specialization mismatch",
        )

    def format(self) -> str:
        return f"grad_enabled == {self.expected}"


@dataclass(frozen=True)
class GuardSetResult:
    results: tuple[GuardResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> tuple[GuardResult, ...]:
        return tuple(result for result in self.results if not result.passed)

    def format(self) -> str:
        return "\n".join(result.format() for result in self.results)


class GuardSet:
    """Deterministic ordered assumptions attached to one cached Graph."""

    def __init__(self, guards: tuple[Guard, ...]) -> None:
        self.guards = tuple(guards)

    @classmethod
    def from_signature(cls, signature: InputSignature) -> "GuardSet":
        guards: list[Guard] = [InputCountGuard(len(signature.tensors))]
        for index, tensor in enumerate(signature.tensors):
            guards.extend(
                [
                    TensorShapeGuard(index, tensor.shape),
                    TensorDTypeGuard(index, tensor.dtype),
                    TensorDeviceGuard(index, tensor.device),
                    RequiresGradGuard(index, tensor.requires_grad),
                ]
            )
        guards.append(GradModeGuard(signature.grad_enabled))
        return cls(tuple(guards))

    def check(self, signature: InputSignature) -> GuardSetResult:
        return GuardSetResult(tuple(guard.check(signature) for guard in self.guards))

    def format(self) -> str:
        return "\n".join(guard.format() for guard in self.guards)

    def __repr__(self) -> str:
        return f"GuardSet({len(self.guards)} guards)"
