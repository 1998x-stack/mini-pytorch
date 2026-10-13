"""Deterministic in-process ProcessGroup simulation for v0.10.

The goal is to teach collective ordering and completion semantics without
pretending that one Python process is a real networked distributed runtime.
Each rank owns a ProcessGroup view, while all views share one in-process world.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


class CollectiveError(RuntimeError):
    """Raised when simulated collective semantics are violated."""


def _validate_world_size(world_size: int) -> int:
    if not isinstance(world_size, (int, np.integer)):
        raise TypeError("world_size must be an integer")
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be >= 1")
    return world_size


def _validate_rank(rank: int, world_size: int) -> int:
    if not isinstance(rank, (int, np.integer)):
        raise TypeError("rank must be an integer")
    rank = int(rank)
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must satisfy 0 <= rank < world_size; got rank={rank}, world_size={world_size}")
    return rank


@dataclass
class _CollectiveSlot:
    op: str
    shape: tuple[int, ...]
    dtype: np.dtype
    contributions: Dict[int, np.ndarray] = field(default_factory=dict)
    result: np.ndarray | None = None
    readers: set[int] = field(default_factory=set)


class _InProcessWorld:
    """Shared collective state for all rank-local ProcessGroup views."""

    def __init__(self, world_size: int) -> None:
        self.world_size = _validate_world_size(world_size)
        self._slots: dict[int, _CollectiveSlot] = {}

    def submit(self, rank: int, sequence: int, value, op: str) -> "CollectiveWork":
        rank = _validate_rank(rank, self.world_size)
        if op not in {"sum", "mean"}:
            raise ValueError("all_reduce op must be 'sum' or 'mean'")

        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("all_reduce expects a numeric ndarray-like value")
        if op == "mean" and not np.issubdtype(array.dtype, np.floating):
            raise TypeError("mean all_reduce currently requires floating-point values")

        shape = tuple(int(dim) for dim in array.shape)
        dtype = np.dtype(array.dtype)
        slot = self._slots.get(sequence)
        if slot is None:
            slot = _CollectiveSlot(op=op, shape=shape, dtype=dtype)
            self._slots[sequence] = slot
        else:
            if slot.op != op:
                raise CollectiveError(
                    f"collective sequence {sequence} operation mismatch: expected {slot.op!r}, got {op!r}"
                )
            if slot.shape != shape:
                raise CollectiveError(
                    f"collective sequence {sequence} shape mismatch: expected {slot.shape}, got {shape}"
                )
            if slot.dtype != dtype:
                raise CollectiveError(
                    f"collective sequence {sequence} dtype mismatch: expected {slot.dtype}, got {dtype}"
                )

        if rank in slot.contributions:
            raise CollectiveError(
                f"rank {rank} submitted collective sequence {sequence} more than once"
            )

        slot.contributions[rank] = np.array(array, copy=True)
        if len(slot.contributions) == self.world_size:
            ordered = [slot.contributions[r] for r in range(self.world_size)]
            total = np.sum(np.stack(ordered, axis=0), axis=0)
            slot.result = total if op == "sum" else total / float(self.world_size)

        return CollectiveWork(self, sequence, rank)

    def is_complete(self, sequence: int) -> bool:
        slot = self._slots.get(sequence)
        return slot is not None and slot.result is not None

    def missing_ranks(self, sequence: int) -> tuple[int, ...]:
        slot = self._slots.get(sequence)
        present = set() if slot is None else set(slot.contributions)
        return tuple(rank for rank in range(self.world_size) if rank not in present)

    def result(self, sequence: int, rank: int) -> np.ndarray:
        rank = _validate_rank(rank, self.world_size)
        slot = self._slots.get(sequence)
        if slot is None or slot.result is None:
            raise CollectiveError(
                f"collective sequence {sequence} is incomplete; missing ranks={list(self.missing_ranks(sequence))}. "
                "The in-process simulator never blocks: submit every rank first, then read Work.result()."
            )
        result = np.array(slot.result, copy=True)
        slot.readers.add(rank)
        if len(slot.readers) == self.world_size:
            # Every rank-local Work caches its first result, so the shared world
            # no longer needs contributions/result payloads for this sequence.
            del self._slots[sequence]
        return result


class CollectiveWork:
    """Handle for one rank's asynchronous simulated collective submission."""

    def __init__(self, world: _InProcessWorld, sequence: int, rank: int) -> None:
        self._world = world
        self.sequence = int(sequence)
        self.rank = int(rank)
        self._cached_result: np.ndarray | None = None

    @property
    def is_completed(self) -> bool:
        return self._cached_result is not None or self._world.is_complete(self.sequence)

    def result(self) -> np.ndarray:
        if self._cached_result is None:
            self._cached_result = self._world.result(self.sequence, self.rank)
        return np.array(self._cached_result, copy=True)

    def wait(self) -> np.ndarray:
        """Return the result if complete; never block in the sequential simulator."""
        return self.result()


class ProcessGroup:
    """Rank-local view over a shared in-process collective world."""

    def __init__(self, world: _InProcessWorld, rank: int) -> None:
        if not isinstance(world, _InProcessWorld):
            raise TypeError("world must be an in-process collective world")
        self._world = world
        self.rank = _validate_rank(rank, world.world_size)
        self.world_size = world.world_size
        self._next_sequence = 0

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def same_world(self, other: "ProcessGroup") -> bool:
        return isinstance(other, ProcessGroup) and self._world is other._world

    def all_reduce(self, value, *, op: str = "mean") -> CollectiveWork:
        sequence = self._next_sequence
        work = self._world.submit(self.rank, sequence, value, op)
        self._next_sequence += 1
        return work

    def __repr__(self) -> str:
        return f"ProcessGroup(rank={self.rank}, world_size={self.world_size}, next_sequence={self._next_sequence})"


def create_process_group(world_size: int) -> tuple[ProcessGroup, ...]:
    """Create one rank-local ProcessGroup view per simulated rank."""
    world = _InProcessWorld(world_size)
    return tuple(ProcessGroup(world, rank) for rank in range(world.world_size))
