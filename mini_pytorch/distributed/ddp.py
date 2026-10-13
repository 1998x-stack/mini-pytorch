"""Educational DistributedDataParallel wrapper and synchronization barrier."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..nn.module import Module
from .process_group import ProcessGroup
from .reducer import Reducer, ReducerError


class DistributedError(RuntimeError):
    """Raised for invalid replica-group synchronization."""


class DistributedDataParallel(Module):
    """Wrap one Module replica and stage Parameter gradients through hooks.

    The wrapper does not synchronize inside backward. Hooks only record that a
    complete leaf gradient is ready. ``synchronize_ddp`` is the explicit
    single-process barrier that launches matching collectives across replicas.
    """

    def __init__(
        self,
        module: Module,
        process_group: ProcessGroup,
        *,
        bucket_cap_numel: int = 1024,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not isinstance(module, Module):
            raise TypeError("module must be an nn.Module")
        if not isinstance(process_group, ProcessGroup):
            raise TypeError("process_group must be a ProcessGroup")
        self.module = module
        object.__setattr__(self, "process_group", process_group)
        reducer = Reducer(
            module.named_parameters(),
            process_group,
            bucket_cap_numel=bucket_cap_numel,
            reduction=reduction,
        )
        object.__setattr__(self, "reducer", reducer)
        handles = []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue

            def hook(grad, *, _name=name, _reducer=reducer):
                _reducer.mark_variable_ready(_name, grad)
                return None

            handles.append(parameter.register_hook(hook))
        object.__setattr__(self, "_ddp_hook_handles", handles)

    @property
    def rank(self) -> int:
        return self.process_group.rank

    @property
    def world_size(self) -> int:
        return self.process_group.world_size

    def forward(self, *args, **kwargs):
        self.reducer.prepare_for_backward()
        return self.module(*args, **kwargs)

    def close(self) -> None:
        for handle in self._ddp_hook_handles:
            handle.remove()
        self._ddp_hook_handles.clear()

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, world_size={self.world_size}, "
            f"buckets={len(self.reducer.buckets)}, reduction={self.reducer.reduction!r}"
        )


def _validated_replicas(replicas: Iterable[DistributedDataParallel]) -> list[DistributedDataParallel]:
    replicas = list(replicas)
    if not replicas:
        raise DistributedError("at least one DDP replica is required")
    if not all(isinstance(replica, DistributedDataParallel) for replica in replicas):
        raise TypeError("all replicas must be DistributedDataParallel objects")

    first = replicas[0]
    world_size = first.world_size
    if len(replicas) != world_size:
        raise DistributedError(
            f"expected exactly world_size={world_size} replicas, got {len(replicas)}"
        )
    if any(replica.world_size != world_size for replica in replicas):
        raise DistributedError("replicas disagree on world_size")
    if any(not first.process_group.same_world(replica.process_group) for replica in replicas[1:]):
        raise DistributedError("replicas do not belong to the same ProcessGroup world")

    ranks = [replica.rank for replica in replicas]
    if len(set(ranks)) != len(ranks):
        raise DistributedError(f"duplicate DDP ranks={ranks}")
    expected = list(range(world_size))
    if sorted(ranks) != expected:
        raise DistributedError(f"DDP ranks must cover {expected}; got {sorted(ranks)}")

    ordered = sorted(replicas, key=lambda replica: replica.rank)
    signature = ordered[0].reducer.bucket_signature
    for replica in ordered[1:]:
        if replica.reducer.bucket_signature != signature:
            raise DistributedError(
                f"rank {replica.rank} bucket/parameter layout does not match rank 0"
            )
    return ordered


def synchronize_ddp(replicas: Iterable[DistributedDataParallel]) -> None:
    """Explicit two-phase synchronization barrier for sequential simulated ranks."""
    ordered = _validated_replicas(replicas)
    not_ready = {
        replica.rank: list(replica.reducer.missing_parameter_names)
        for replica in ordered
        if not replica.reducer.all_ready
    }
    if not_ready:
        raise DistributedError(f"cannot synchronize because gradients are not ready: {not_ready}")

    # Phase 1: every rank submits every bucket collective. No rank waits here.
    for replica in ordered:
        replica.reducer.launch_collectives()

    # Phase 2: all matching submissions now exist, so Work.result() is complete.
    for replica in ordered:
        replica.reducer.finish_collectives()


def broadcast_module_state(
    replicas: Iterable[DistributedDataParallel],
    *,
    src: int = 0,
) -> None:
    """Copy Parameter/Buffer values from one replica to all peers before training.

    Real DDP performs initial module-state synchronization through distributed
    collectives. The in-process educational version makes the state-copy phase
    explicit while keeping ProcessGroup focused on gradient collectives.
    """
    ordered = _validated_replicas(replicas)
    if not isinstance(src, (int, np.integer)) or not 0 <= int(src) < len(ordered):
        raise ValueError("src must be a valid rank")
    source = next(replica for replica in ordered if replica.rank == int(src))
    state = source.module.state_dict()
    for replica in ordered:
        if replica is source:
            continue
        replica.module.load_state_dict(state, strict=True)
