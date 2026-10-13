# Agents.md — mini-pytorch v0.10 Development Contract

> Read this file before modifying branch `v0.10`.

## Branch identity

- Branch: `v0.10`
- Previous version: `v0.9`
- Primary mental model: **Distributed / DDP Concepts**
- Status: active development
- Target package version: `0.10.0`
- Previous validated baseline: `140/140 tests passed`

## 1. Version question

> If several replicas start from the same parameters but compute different local gradients, how are gradient-ready signals, buckets and collectives coordinated so all replicas apply the same optimizer update?

## 2. Required architecture

```text
Dataset
  ↓
DistributedSampler
  ↓
rank-local batch
  ↓
Module replica
  ↓
AutogradEngine
  ↓
leaf Parameter hooks
  ↓
Reducer
  ↓
GradientBucket(s)
  ↓
ProcessGroup.all_reduce
  ↓
averaged Parameter.grad
  ↓
Optimizer.step
```

Layer ownership is mandatory:

- `DistributedSampler` owns index partitioning;
- `ProcessGroup` owns collective semantics;
- `Reducer` owns ready-state, flatten/unflatten and bucket logic;
- `DistributedDataParallel` owns Module wrapping and hook registration;
- Optimizer remains unaware of distributed synchronization;
- Autograd formulas remain unchanged.

## 3. Simulation model

This branch must run on ordinary single-process CPU GitHub Actions.

Do not pretend that sequential ranks are concurrent network workers.

The required execution model is:

```text
rank 0 backward  → stage grads
rank 1 backward  → stage grads
...
explicit synchronization barrier
  ↓
launch matching collectives for every rank
  ↓
complete in-process all-reduce
  ↓
write averaged grads back
```

A rank-local hook must never block waiting for another rank.

## 4. ProcessGroup rules

- `world_size >= 1`;
- `0 <= rank < world_size`;
- rank-local ProcessGroup views share one in-process world;
- each collective has a monotonically increasing sequence number per rank;
- matching sequence numbers must agree on operation, shape and dtype;
- `sum` and `mean` reductions are explicit;
- duplicate/missing/mismatched participation must fail clearly;
- a Work handle queried before all ranks submit raises an incomplete-collective error instead of deadlocking.

## 5. DistributedSampler rules

- subclass/reuse the existing Sampler abstraction;
- partition indices by rank after deterministic global ordering;
- `set_epoch(epoch)` changes shuffled ordering deterministically;
- `drop_last=False` pads to a divisible total size;
- `drop_last=True` truncates to a divisible total size;
- sampler does not fetch samples and DataLoader does not contain rank-specific logic.

## 6. Reducer rules

Reducer receives the wrapped module's named Parameters in deterministic order.

Required behavior:

- build deterministic buckets using a documented `bucket_cap_numel`;
- register one leaf gradient hook per parameter through DDP;
- stage the complete local gradient seen by that hook;
- reject duplicate readiness in one iteration;
- require all expected parameters to become ready before synchronization;
- flatten using bucket entry order;
- all-reduce the flat gradient buffer;
- unflatten by recorded offset/shape;
- replace local `Parameter.grad` with the reduced/averaged value;
- preserve dtype/shape/device identity of Parameter objects.

Unused-parameter discovery is a non-goal; an expected parameter that never becomes ready must produce an explicit error.

## 7. DDP wrapper rules

`DistributedDataParallel` should behave as an educational Module wrapper:

- delegate `forward` to the wrapped Module;
- keep the wrapped Module registered as a child;
- expose rank/world-size through its ProcessGroup;
- prepare Reducer state at the start of a new forward/backward iteration;
- register hooks once at construction;
- provide hook removal/cleanup where practical;
- synchronization is explicit and separate from `forward`, `backward` and optimizer code.

## 8. Reference-correctness requirement

The central integration proof must compare:

```text
replica 0 local batch gradient
replica 1 local batch gradient
        ↓ average
DDP synchronized gradient
```

with:

```text
one reference model
concatenated global batch
mean-reduced loss
        ↓
reference gradient
```

After identical optimizer steps:

```text
replica 0 parameters
== replica 1 parameters
== global-batch reference parameters
```

Use equal local batch sizes for this v0.10 proof so averaging semantics are unambiguous.

## 9. Historical invariants

All v0.9 and earlier behavior must remain intact:

- Tensor/Storage/view/version semantics;
- Dispatcher and CPU/EDU backend routing;
- Autograd Engine graph lifetime and hooks;
- Module/Optimizer/Scheduler state;
- DataLoader behavior;
- framework RNG and deterministic checkpoint resume;
- CNN operators.

Distributed code must not be imported by low-level kernels or Autograd Engine internals.

## 10. Testing contract

Add `tests/test_distributed.py` with at least:

- rank/world-size validation;
- all-reduce sum/mean;
- incomplete Work failure;
- collective order/shape/dtype mismatch;
- sampler partition/padding/drop_last/set_epoch;
- reducer bucket construction;
- hook readiness;
- missing/unused parameter synchronization failure;
- replica synchronized-grad equality;
- DDP vs global-batch reference update equivalence;
- EDU parameter placement preservation if exercised.

Run the full suite:

```bash
python -m unittest discover -s tests -v
```

Exact final branch-head GitHub Actions must be green before freeze.

## 11. Documentation contract

Before freeze update:

```text
README.md
version.md
Agents.md
docs/DISTRIBUTED_DDP.md
examples/13_ddp_simulation.py
MANIFEST.txt
```

README must be the version-specific learning entry point.

## 12. Explicit non-goals

Do not add in v0.10:

- actual multiprocessing/network sockets;
- NCCL/Gloo/MPI;
- CUDA collective execution;
- FSDP/ZeRO;
- sharded optimizer state;
- distributed checkpointing;
- `no_sync()`;
- unused-parameter graph traversal;
- join context for uneven inputs;
- production collective retry/failure handling;
- compiler graph capture.

## 13. Definition of Done

v0.10 can be frozen only after:

- ProcessGroup simulation is complete and tested;
- DistributedSampler is complete and tested;
- DDP/Reducer/hooks/buckets are complete and tested;
- explicit synchronization barrier is demonstrated;
- global-batch reference equivalence passes;
- full historical suite passes;
- educational docs/example are complete;
- package version is `0.10.0`;
- exact final branch-head CI is green.
