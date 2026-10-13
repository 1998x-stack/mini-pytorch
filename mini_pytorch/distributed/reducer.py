"""DDP-style gradient readiness tracking and bucket reduction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..nn.module import Parameter
from .process_group import CollectiveWork, ProcessGroup


class ReducerError(RuntimeError):
    """Raised when Reducer iteration/bucket invariants are violated."""


@dataclass(frozen=True)
class BucketEntry:
    name: str
    parameter: Parameter
    offset: int
    numel: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class GradientBucket:
    index: int
    entries: tuple[BucketEntry, ...]
    total_numel: int
    dtype: np.dtype
    device: str

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)


class Reducer:
    """Stage rank-local leaf gradients, bucket them, then launch collectives.

    Buckets use reverse parameter-registration order, mirroring the common DDP
    intuition that later layers tend to become gradient-ready first. Different
    dtype/device parameters always start separate buckets.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, Parameter]],
        process_group: ProcessGroup,
        *,
        bucket_cap_numel: int = 1024,
        reduction: str = "mean",
    ) -> None:
        if not isinstance(process_group, ProcessGroup):
            raise TypeError("process_group must be a ProcessGroup")
        if not isinstance(bucket_cap_numel, (int, np.integer)) or int(bucket_cap_numel) <= 0:
            raise ValueError("bucket_cap_numel must be a positive integer")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")

        params = [(str(name), param) for name, param in named_parameters if param.requires_grad]
        if not params:
            raise ValueError("DDP Reducer requires at least one trainable Parameter")
        if not all(isinstance(param, Parameter) for _, param in params):
            raise TypeError("Reducer expects nn.Parameter objects")
        ids = [id(param) for _, param in params]
        if len(ids) != len(set(ids)):
            raise ReducerError("shared/duplicated Parameter objects are not supported in v0.10 Reducer")
        names = [name for name, _ in params]
        if len(names) != len(set(names)):
            raise ReducerError("parameter names must be unique")

        self.process_group = process_group
        self.bucket_cap_numel = int(bucket_cap_numel)
        self.reduction = reduction
        self._parameters = dict(params)
        self.buckets = self._build_buckets(list(reversed(params)))
        self._staged: dict[str, np.ndarray] = {}
        self._ready: set[str] = set()
        self._works: tuple[CollectiveWork, ...] | None = None
        self._active = False
        self.iteration = 0

    def _build_buckets(self, reversed_params: list[tuple[str, Parameter]]) -> tuple[GradientBucket, ...]:
        buckets: list[GradientBucket] = []
        current: list[tuple[str, Parameter]] = []
        current_numel = 0
        current_dtype: np.dtype | None = None
        current_device: str | None = None

        def flush() -> None:
            nonlocal current, current_numel, current_dtype, current_device
            if not current:
                return
            offset = 0
            entries: list[BucketEntry] = []
            for name, parameter in current:
                numel = parameter.numel()
                entries.append(
                    BucketEntry(
                        name=name,
                        parameter=parameter,
                        offset=offset,
                        numel=numel,
                        shape=tuple(parameter.shape),
                    )
                )
                offset += numel
            buckets.append(
                GradientBucket(
                    index=len(buckets),
                    entries=tuple(entries),
                    total_numel=offset,
                    dtype=np.dtype(current_dtype),
                    device=str(current_device),
                )
            )
            current = []
            current_numel = 0
            current_dtype = None
            current_device = None

        for name, parameter in reversed_params:
            numel = parameter.numel()
            dtype = np.dtype(parameter.dtype)
            device = str(parameter.device)
            incompatible = current and (dtype != current_dtype or device != current_device)
            exceeds = current and current_numel + numel > self.bucket_cap_numel
            if incompatible or exceeds:
                flush()
            if not current:
                current_dtype = dtype
                current_device = device
            current.append((name, parameter))
            current_numel += numel
        flush()
        return tuple(buckets)

    @property
    def rank(self) -> int:
        return self.process_group.rank

    @property
    def world_size(self) -> int:
        return self.process_group.world_size

    @property
    def all_ready(self) -> bool:
        return len(self._ready) == len(self._parameters)

    @property
    def ready_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name in self._parameters if name in self._ready)

    @property
    def missing_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name in self._parameters if name not in self._ready)

    @property
    def bucket_signature(self) -> tuple:
        return tuple(
            (
                bucket.index,
                bucket.dtype.str,
                bucket.device,
                tuple((entry.name, entry.shape, entry.numel, entry.offset) for entry in bucket.entries),
            )
            for bucket in self.buckets
        )

    def prepare_for_backward(self) -> None:
        if self._active and self._ready:
            raise ReducerError(
                "previous DDP iteration has staged gradients that were not synchronized"
            )
        stale = [name for name, parameter in self._parameters.items() if parameter.grad is not None]
        if stale:
            raise ReducerError(
                "DDP v0.10 requires gradients to be cleared before the next forward; "
                f"stale gradients={stale}"
            )
        self._staged.clear()
        self._ready.clear()
        self._works = None
        self._active = True
        self.iteration += 1

    def mark_variable_ready(self, name: str, grad: np.ndarray) -> None:
        if not self._active:
            raise ReducerError(
                "gradient hook fired before Reducer.prepare_for_backward(); call the DDP wrapper forward"
            )
        if name not in self._parameters:
            raise ReducerError(f"unknown reducer parameter {name!r}")
        if name in self._ready:
            raise ReducerError(f"parameter {name!r} became ready more than once in one iteration")
        parameter = self._parameters[name]
        value = np.asarray(grad, dtype=parameter.data.dtype)
        if value.shape != parameter.shape:
            raise ReducerError(
                f"gradient shape for {name!r} is {value.shape}; expected {parameter.shape}"
            )
        self._staged[name] = np.array(value, copy=True)
        self._ready.add(name)

    def _flatten_bucket(self, bucket: GradientBucket) -> np.ndarray:
        chunks = [self._staged[entry.name].reshape(-1) for entry in bucket.entries]
        if not chunks:
            return np.empty((0,), dtype=bucket.dtype)
        flat = np.concatenate(chunks)
        if flat.size != bucket.total_numel:
            raise ReducerError("bucket flatten size mismatch")
        return flat.astype(bucket.dtype, copy=False)

    def launch_collectives(self) -> tuple[CollectiveWork, ...]:
        if not self._active:
            raise ReducerError("Reducer is not prepared for a backward iteration")
        if not self.all_ready:
            raise ReducerError(
                f"cannot synchronize rank {self.rank}: gradients not ready for {list(self.missing_parameter_names)}"
            )
        if self._works is not None:
            raise ReducerError("collectives already launched for this iteration")
        self._works = tuple(
            self.process_group.all_reduce(self._flatten_bucket(bucket), op=self.reduction)
            for bucket in self.buckets
        )
        return self._works

    def finish_collectives(self) -> None:
        if self._works is None:
            raise ReducerError("collectives have not been launched")
        if len(self._works) != len(self.buckets):
            raise ReducerError("collective work count does not match bucket count")

        for bucket, work in zip(self.buckets, self._works):
            reduced = np.asarray(work.result(), dtype=bucket.dtype)
            if reduced.shape != (bucket.total_numel,):
                raise ReducerError(
                    f"reduced bucket {bucket.index} has shape {reduced.shape}; expected {(bucket.total_numel,)}"
                )
            for entry in bucket.entries:
                start = entry.offset
                stop = start + entry.numel
                grad = reduced[start:stop].reshape(entry.shape)
                entry.parameter.grad = np.array(grad, copy=True)

        self._active = False
        self._staged.clear()
        self._ready.clear()
        self._works = None
