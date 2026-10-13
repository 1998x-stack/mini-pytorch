"""Educational guarded compiled-function wrapper for mini-pytorch v0.12."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .cache import EntryCheck, GraphCache
from .capture import GraphBreakError, capture
from .interpreter import GraphInterpreter
from .signature import InputSignature


@dataclass(frozen=True)
class CompilationStats:
    cache_entries: int
    compile_count: int
    recompile_count: int
    hit_count: int
    miss_count: int
    graph_break_count: int


@dataclass(frozen=True)
class CompilationExplanation:
    signature: InputSignature
    checks: tuple[EntryCheck, ...]

    @property
    def matching_compile_id(self) -> int | None:
        for check in self.checks:
            if check.passed:
                return check.compile_id
        return None

    def format(self) -> str:
        lines = ["runtime signature:", self.signature.format()]
        if not self.checks:
            lines.append("cache: EMPTY")
            return "\n".join(lines)
        for check in self.checks:
            lines.append(f"entry {check.compile_id}: {'PASS' if check.passed else 'FAIL'}")
            failures = check.result.failures
            if not failures:
                lines.append("  all guards passed")
            else:
                lines.extend(f"  {failure.format()}" for failure in failures)
        return "\n".join(lines)


class CompiledFunction:
    """One Python callable with zero or more guarded Graph specializations."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise TypeError("CompiledFunction expects a callable")
        self.fn = fn
        self.cache = GraphCache()
        self.compile_count = 0
        self.graph_break_count = 0

    def __call__(self, *inputs: "Tensor"):
        signature = InputSignature.from_inputs(*inputs)
        lookup = self.cache.lookup(signature)
        if lookup.entry is not None:
            return GraphInterpreter(lookup.entry.graph)(*inputs)

        try:
            captured = capture(self.fn, *inputs)
        except GraphBreakError:
            self.graph_break_count += 1
            raise

        compile_id = self.compile_count
        self.cache.add(
            captured.graph,
            signature,
            compile_id=compile_id,
            metadata={"function": getattr(self.fn, "__name__", type(self.fn).__name__)},
        )
        self.compile_count += 1
        # Capture already ran the function eagerly. Returning its output avoids
        # immediately executing the same graph a second time on a cache miss.
        return captured.eager_output

    def stats(self) -> CompilationStats:
        return CompilationStats(
            cache_entries=len(self.cache),
            compile_count=self.compile_count,
            recompile_count=max(0, self.compile_count - 1),
            hit_count=self.cache.hits,
            miss_count=self.cache.misses,
            graph_break_count=self.graph_break_count,
        )

    def explain(self, *inputs: "Tensor") -> CompilationExplanation:
        signature = InputSignature.from_inputs(*inputs)
        return CompilationExplanation(signature, self.cache.explain(signature))

    def clear_cache(self) -> None:
        self.cache.clear()
        self.compile_count = 0
        self.graph_break_count = 0


def compile_graph(fn: Callable[..., Any]) -> CompiledFunction:
    """Wrap ``fn`` with v0.12 guarded graph specialization and caching."""
    return CompiledFunction(fn)


try:  # pragma: no cover
    from ..tensor import Tensor
except Exception:  # pragma: no cover
    Tensor = object
