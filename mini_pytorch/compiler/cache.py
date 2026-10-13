"""Guarded graph cache for mini-pytorch v0.12."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .guards import GuardSet, GuardSetResult
from .ir import Graph
from .signature import InputSignature


@dataclass
class GraphCacheEntry:
    graph: Graph
    guards: GuardSet
    signature: InputSignature
    compile_id: int
    hit_count: int = 0
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EntryCheck:
    compile_id: int
    result: GuardSetResult

    @property
    def passed(self) -> bool:
        return self.result.passed


@dataclass(frozen=True)
class CacheLookup:
    entry: GraphCacheEntry | None
    checks: tuple[EntryCheck, ...]

    @property
    def hit(self) -> bool:
        return self.entry is not None


class GraphCache:
    """Small ordered cache; linear search keeps guard behavior inspectable."""

    def __init__(self) -> None:
        self.entries: list[GraphCacheEntry] = []
        self.hits = 0
        self.misses = 0

    def lookup(self, signature: InputSignature) -> CacheLookup:
        checks: list[EntryCheck] = []
        for entry in self.entries:
            result = entry.guards.check(signature)
            checks.append(EntryCheck(entry.compile_id, result))
            if result.passed:
                entry.hit_count += 1
                self.hits += 1
                return CacheLookup(entry, tuple(checks))
        self.misses += 1
        return CacheLookup(None, tuple(checks))

    def explain(self, signature: InputSignature) -> tuple[EntryCheck, ...]:
        """Evaluate guards without mutating hit/miss counters."""
        return tuple(
            EntryCheck(entry.compile_id, entry.guards.check(signature))
            for entry in self.entries
        )

    def add(
        self,
        graph: Graph,
        signature: InputSignature,
        *,
        compile_id: int,
        metadata: dict[str, Any] | None = None,
    ) -> GraphCacheEntry:
        guards = GuardSet.from_signature(signature)
        entry = GraphCacheEntry(
            graph=graph,
            guards=guards,
            signature=signature,
            compile_id=int(compile_id),
            metadata=dict(metadata or {}),
        )
        self.entries.append(entry)
        return entry

    def clear(self) -> None:
        self.entries.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self.entries)
