# mini-pytorch v0.10

## 1. Version identity

- **Branch:** `v0.10`
- **Previous version:** `v0.9`
- **Theme:** Distributed / DDP Concepts
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.10.0`
- **Validated regression baseline:** **159/159 tests passed**

v0.10 answers:

> If multiple model replicas compute different local gradients, how are dataset partitioning, gradient-ready hooks, buckets and collective reduction composed so every replica performs an equivalent optimizer update?

## 2. Core difference from v0.9

v0.9 made one training process resumable:

```text
Module + Optimizer + Scheduler + RNG + TrainingState
                         ↓
                    CheckpointState
```

v0.10 adds replicated gradient synchronization:

```text
Dataset
  ↓
DistributedSampler
  ↓
rank-local batches
  ↓
replica forward/backward
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
optimizer.step()
```

The milestone is DDP architecture, not production networking.

## 3. Implemented mechanisms

### Rank / world-size semantics

Added validated rank-local process-group views:

```text
world_size >= 1
0 <= rank < world_size
```

`create_process_group(world_size)` constructs one `ProcessGroup` per simulated rank, all sharing one deterministic in-process world.

### ProcessGroup + CollectiveWork

Added:

```text
ProcessGroup
CollectiveWork
CollectiveError
create_process_group
```

Supported all-reduce operations:

```text
sum
mean
```

Every rank owns a monotonically increasing collective sequence number.

Matching collective sequence numbers must agree on:

- reduction operation;
- shape;
- dtype.

`CollectiveWork.result()` before all ranks have submitted fails clearly instead of blocking. This is an explicit simulation rule: sequential Python ranks cannot safely wait inside a rank-local call for peers that have not executed yet.

### DistributedSampler

Added deterministic rank partitioning:

```text
one global ordering
      ↓
indices[rank : total_size : num_replicas]
```

Supports:

- shuffle;
- fixed seed;
- `set_epoch(epoch)`;
- equal-size padding when `drop_last=False`;
- divisible truncation when `drop_last=True`.

The sampler reuses the existing `Sampler` abstraction and does not fetch samples itself.

### DistributedDataParallel

Added educational `DistributedDataParallel(Module)` wrapper.

The wrapped Module remains an ordinary child Module. DDP registers one Tensor backward hook per trainable leaf Parameter.

The hooks do not modify backward formulas and do not perform blocking communication. They only stage the complete rank-local gradient in the Reducer.

### Reducer

Added:

```text
Reducer
ReducerError
GradientBucket
BucketEntry
```

Reducer owns:

- iteration preparation;
- stale-gradient boundary checks;
- leaf-gradient readiness tracking;
- duplicate readiness rejection;
- deterministic bucket construction;
- flattening;
- ProcessGroup collective launch;
- Work completion;
- unflattening;
- synchronized `.grad` write-back.

### Gradient buckets

Buckets visit trainable Parameters in reverse registration order, matching the teaching intuition that later layers commonly become gradient-ready first during reverse-mode autodiff.

A documented `bucket_cap_numel` controls educational bucket size.

Each bucket records:

```text
index
dtype
device
parameter name
parameter shape
parameter numel
flat offset
```

Different dtype/device parameters are never mixed in the same bucket.

### Explicit synchronization barrier

Because simulated ranks run sequentially in one process, v0.10 uses a two-phase synchronization contract:

```text
Phase 1
rank-local backward
→ hooks stage local gradients

Phase 2
synchronize_ddp(all_replicas)
→ every rank launches matching bucket collectives
→ collectives become complete
→ every reducer reads Work results
→ averaged gradients are unflattened into Parameter.grad
```

Optimizer algorithms remain unchanged and must run after synchronization.

### Initial module-state broadcast helper

Added:

```python
broadcast_module_state(replicas, src=0)
```

It copies Parameters/Buffers from one source replica to peer replicas before training.

The in-process implementation makes this initialization phase explicit rather than pretending to provide a network broadcast backend.

### Unused/stale gradient failures

v0.10 intentionally does not implement unused-parameter graph discovery or `no_sync()` accumulation.

Therefore:

- an expected trainable Parameter that never becomes ready causes synchronization to fail with its name;
- entering the next DDP forward with stale `.grad` values causes an explicit Reducer error;
- the supported iteration lifecycle is `zero_grad → forward → backward → synchronize → optimizer.step`.

### EDU placement preservation

DDP collective payloads are host NumPy gradient values, but synchronized results are written into existing Parameter `.grad` arrays without changing Parameter/Storage device identity.

Tests prove EDU Parameters remain EDU after synchronization.

## 4. Central correctness proof

The central integration test starts three identical `Linear(2, 1)` models.

Distributed path:

```text
rank 0: two samples → local mean MSE → backward
rank 1: two samples → local mean MSE → backward
      ↓
synchronize_ddp
      ↓
average local gradients
      ↓
SGD step on both replicas
```

Reference path:

```text
one model
all four samples
mean MSE
      ↓
backward
      ↓
SGD step
```

CI proves at tight tolerance:

```text
replica 0 parameters
== replica 1 parameters
== global-batch reference parameters
```

This proves more than replica equality: the synchronized update is mathematically equivalent to the intended global-batch update for equal local batch sizes and mean reduction.

## 5. New files

```text
mini_pytorch/distributed/__init__.py
mini_pytorch/distributed/process_group.py
mini_pytorch/distributed/sampler.py
mini_pytorch/distributed/reducer.py
mini_pytorch/distributed/ddp.py

tests/test_distributed.py
docs/DISTRIBUTED_DDP.md
examples/13_ddp_simulation.py
```

Modified:

```text
mini_pytorch/__init__.py
pyproject.toml
README.md
MANIFEST.txt
Agents.md
version.md
```

## 6. Preserved invariants

v0.10 preserves:

- Tensor/Storage/stride/view/version-counter behavior;
- CPU/EDU Device and Dispatcher semantics;
- Autograd Engine graph lifetime/hooks/custom Function behavior;
- Module/Optimizer/Scheduler behavior;
- Dataset/Sampler/DataLoader behavior;
- CNN operators;
- framework RNG;
- v0.9 deterministic checkpoint resume.

Distributed synchronization remains above Autograd numerical formulas and backend kernels.

## 7. Validation

Validated implementation commit:

```text
52a3e54085b06664d341e7297bcfe9b7f0d6466f
```

GitHub Actions environment:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

Result:

```text
Ran 159 tests in 0.983s
OK
```

This is the v0.9 baseline of 140 tests plus 19 dedicated v0.10 distributed tests.

## 8. v0.10-specific test coverage

- rank/world-size creation and validation;
- async educational Work completion;
- mean all-reduce;
- sum all-reduce;
- operation mismatch;
- shape mismatch;
- dtype mismatch;
- DistributedSampler divisible partition;
- padding and `drop_last`;
- deterministic `set_epoch` shuffle;
- invalid rank;
- deterministic reversed bucket layout;
- leaf gradient-ready hook staging;
- synchronized local-gradient averaging;
- missing-rank barrier failure;
- unused-parameter failure;
- stale-gradient iteration boundary;
- initial module-state broadcast;
- EDU placement preservation;
- DDP-vs-global-batch optimizer-step equivalence.

## 9. Recommended reading order

```text
mini_pytorch/distributed/process_group.py
        ↓
mini_pytorch/distributed/sampler.py
        ↓
mini_pytorch/distributed/reducer.py
        ↓
mini_pytorch/distributed/ddp.py
        ↓
tests/test_distributed.py
        ↓
docs/DISTRIBUTED_DDP.md
        ↓
examples/13_ddp_simulation.py
```

## 10. Explicit non-goals

v0.10 intentionally does not implement:

- real multiprocessing or multi-node execution;
- TCP/shared-memory transport;
- NCCL/Gloo/MPI;
- CUDA collectives/streams/events;
- real backward/communication overlap;
- `no_sync()`;
- unused-parameter graph traversal;
- uneven-input join behavior;
- FSDP/ZeRO;
- sharded optimizer state;
- distributed checkpointing;
- production collective fault tolerance.

## 11. Definition of Done

- [x] rank/world-size validation
- [x] in-process ProcessGroup + Work
- [x] sum/mean all-reduce
- [x] collective compatibility failures
- [x] DistributedSampler + set_epoch
- [x] DDP wrapper + leaf hooks
- [x] Reducer ready-state tracking
- [x] deterministic gradient buckets
- [x] explicit synchronization barrier
- [x] replica-consistent gradients
- [x] global-batch reference equivalence
- [x] EDU placement preservation
- [x] full historical regression green
- [x] README/docs/example complete
- [x] package version `0.10.0`
- [x] branch frozen pending exact frozen-head CI/integration

## 12. Next version

`v0.11` should focus on **Graph IR / Eager-to-Graph Capture**:

```text
eager Tensor operators
      ↓
capture/tracing boundary
      ↓
Graph / Node / Value IR
      ↓
graph interpreter
      ↓
eager-vs-graph equivalence
```

The next central question is:

> How can an eager sequence of operator calls be represented as explicit data that can be inspected, replayed and later optimized?
