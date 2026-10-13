# mini-pytorch v0.10

## 1. Version identity

- **Branch:** `v0.10`
- **Previous version:** `v0.9`
- **Theme:** Distributed / DDP Concepts
- **Status:** active development
- **Target package version:** `0.10.0`
- **Previous validated baseline:** `140/140 tests passed`

v0.10 asks:

> If multiple model replicas compute different local gradients, how are data partitioning, gradient-ready hooks, buckets and collective reduction composed so every replica performs the same optimizer update?

## 2. Core delta from v0.9

v0.9 made one training process checkpointable:

```text
Module + Optimizer + Scheduler + RNG + TrainingState
                         ↓
                    CheckpointState
```

v0.10 adds replica synchronization above the existing Autograd Engine:

```text
global Dataset
      ↓
DistributedSampler
      ↓
rank-local batches
      ↓
replica forward/backward
      ↓
parameter gradient hooks
      ↓
Reducer
      ↓
gradient buckets
      ↓
ProcessGroup.all_reduce
      ↓
averaged Parameter.grad
      ↓
optimizer.step()
```

## 3. Planned mechanisms

### Rank / world size

- validated rank in `[0, world_size)`;
- explicit `world_size >= 1`;
- one rank-local ProcessGroup view per simulated replica.

### In-process ProcessGroup

A shared deterministic world will simulate collectives on ordinary CPU CI.

The rank-local API will submit an asynchronous educational all-reduce and return a Work handle. Because ranks execute sequentially in one Python process, `Work.result()` before all ranks submit must fail clearly rather than block forever.

Collective sequence, operation, shape and dtype compatibility are validated.

### DistributedSampler

- deterministic rank partitioning;
- `set_epoch(epoch)` reshuffling;
- padding when `drop_last=False`;
- truncation when `drop_last=True`;
- no DataLoader-internal distributed special case.

### DistributedDataParallel wrapper

- wraps an existing Module;
- registers hooks on leaf Parameters;
- delegates forward to the wrapped Module;
- Reducer owns gradient synchronization mechanics.

### Reducer and buckets

- deterministic parameter ordering;
- bucket construction by a small `bucket_cap_numel` teaching limit;
- leaf hooks stage complete local gradients;
- flatten/unflatten preserves parameter shape and order;
- explicit ready-state validation;
- reduced gradients overwrite local `Parameter.grad` before optimizer step.

### Explicit synchronization barrier

Sequential simulated ranks must not block inside backward hooks.

v0.10 therefore uses two phases:

```text
Phase 1
rank-local backward
→ hook stages local grads

Phase 2
synchronize_ddp([...])
→ every rank launches matching collectives
→ collectives complete
→ reducers unflatten averaged gradients
→ optimizer.step()
```

### Reference equivalence

For equal local batch sizes and mean-reduced loss:

```text
average(local gradients across ranks)
==
gradient(global concatenated batch)
```

An integration test must compare DDP replicas after synchronization/update with a single-model global-batch reference.

## 4. Explicit non-goals

v0.10 does not implement:

- multiprocessing or multi-node networking;
- NCCL/Gloo/MPI;
- CUDA streams/events;
- overlap of real communication with backward compute;
- `no_sync()` gradient accumulation;
- unused-parameter discovery;
- join context for uneven inputs;
- sharded parameters/optimizer state;
- production collective fault tolerance.

## 5. Proposed source layout

```text
mini_pytorch/distributed/
├── __init__.py
├── process_group.py
├── sampler.py
├── reducer.py
└── ddp.py

tests/test_distributed.py
docs/DISTRIBUTED_DDP.md
examples/13_ddp_simulation.py
```

## 6. Definition of Done

Before freeze:

- [ ] rank/world-size validation
- [ ] in-process ProcessGroup + Work
- [ ] sum/mean all-reduce semantics
- [ ] collective order/shape/dtype failure paths
- [ ] DistributedSampler + set_epoch
- [ ] DDP wrapper parameter hooks
- [ ] Reducer readiness tracking
- [ ] deterministic gradient buckets
- [ ] explicit synchronization barrier
- [ ] replica-consistent gradients
- [ ] global-batch reference equivalence
- [ ] CPU/EDU placement remains valid where applicable
- [ ] full historical regression stays green
- [ ] README/docs/example complete
- [ ] exact final branch-head CI green
- [ ] branch frozen and integrated into main

## 7. Learning boundary

v0.10 teaches the architecture of replicated data-parallel gradient synchronization. The in-process world is a transport simulator, not a claim of real distributed execution.