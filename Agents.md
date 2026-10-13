# Agents.md — mini-pytorch v0.10 Frozen Branch Contract

> Read this file before modifying the frozen `v0.10` branch.

## Branch identity

- Branch: `v0.10`
- Previous version: `v0.9`
- Primary mental model: **Distributed / DDP Concepts**
- Status: **frozen historical learning snapshot**
- Package version: `0.10.0`
- Validated baseline: `159/159 tests passed`

## 1. Frozen learning model

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
leaf gradient hooks
  ↓
Reducer
  ↓
GradientBucket(s)
  ↓
ProcessGroup.all_reduce(mean/sum)
  ↓
synchronized Parameter.grad
  ↓
Optimizer.step
```

## 2. Historical branch rule

This branch is frozen.

Allowed only:

- documentation corrections;
- tests clarifying promised v0.10 behavior;
- narrowly scoped correctness fixes inside the distributed/DDP boundary.

Forbidden:

- v0.11 Graph IR/compiler work;
- production networking/NCCL/Gloo/MPI;
- FSDP/ZeRO;
- unrelated NN/operator expansion;
- backporting later architectural milestones.

## 3. Frozen ProcessGroup semantics

- rank/world-size are explicit and validated;
- rank-local ProcessGroup views share one in-process world;
- each rank advances a monotonically increasing collective sequence;
- matching sequences must agree on op/shape/dtype;
- supported reductions are `sum` and `mean`;
- Work handles are incomplete until every rank submits;
- incomplete Work access fails instead of blocking;
- the implementation is a deterministic transport simulator, not real distributed execution.

Do not replace this with blocking rank-local waits inside sequential CI.

## 4. Frozen DistributedSampler semantics

- global deterministic ordering is created before partitioning;
- rank partition is a strided slice of the global order;
- `set_epoch` changes shuffle deterministically using seed + epoch;
- `drop_last=False` pads to equal rank sample counts;
- `drop_last=True` truncates to a divisible total size;
- the sampler selects indices only and does not fetch data.

## 5. Frozen Reducer semantics

Reducer owns:

- iteration readiness state;
- trainable Parameter expectations;
- reverse-registration bucket construction;
- dtype/device bucket separation;
- flatten offsets;
- collective launch;
- Work completion;
- unflatten/write-back.

Every expected trainable Parameter must become ready exactly once per supported iteration.

Unused-parameter graph discovery is outside v0.10.

## 6. Frozen DDP execution contract

Supported training lifecycle:

```text
optimizer.zero_grad()
      ↓
DDP forward
      ↓
backward
      ↓
synchronize_ddp(all replicas)
      ↓
optimizer.step()
```

Backward hooks only stage gradients; they never block on peers.

`synchronize_ddp` is the explicit two-phase barrier:

```text
launch all rank collectives
      ↓
all collectives complete
      ↓
finish/unflatten all ranks
```

Do not move distributed synchronization into Tensor math, Function backward formulas, backend kernels or optimizer algorithms.

## 7. Frozen bucket invariant

Bucket signature includes:

```text
index
dtype
device
parameter names
shapes
numel
offsets
```

All replicas in one synchronization group must have the same bucket signature.

Different dtype/device Parameters do not share a bucket.

## 8. Frozen reduction semantics

Default DDP reduction is `mean` across ranks.

The reference-equivalence guarantee assumes equal local batch sizes and local losses reduced by mean.

Uneven local sample counts are not claimed to equal sample-weighted global-batch training.

## 9. Frozen correctness proof

The release integration proof compares:

```text
two rank-local mean gradients
      ↓
mean all-reduce
      ↓
SGD step on both replicas
```

against:

```text
one global concatenated batch
      ↓
mean loss/backward
      ↓
SGD step
```

Validated result:

```text
replica0 parameters
== replica1 parameters
== reference parameters
```

Do not weaken or remove this proof.

## 10. Device invariant

Collectives operate on NumPy gradient payloads in the simulator, but Parameter objects and Storage placement are not recreated during synchronization.

EDU Parameters must remain EDU after gradient synchronization.

## 11. v0.9 runtime regression

The deterministic checkpoint-resume test from v0.9 remains part of the full regression suite and must stay green.

Distributed work must not alter framework RNG/checkpoint guarantees.

## 12. Validation baseline

Validated implementation commit:

```text
52a3e54085b06664d341e7297bcfe9b7f0d6466f
```

GitHub Actions:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
Ran 159 tests in 0.983s
OK
```

Any future v0.10 fix must rerun the complete suite and pass CI on the exact modified branch HEAD.

## 13. Documentation map

```text
README.md
version.md
mini_pytorch/distributed/process_group.py
mini_pytorch/distributed/sampler.py
mini_pytorch/distributed/reducer.py
mini_pytorch/distributed/ddp.py
tests/test_distributed.py
docs/DISTRIBUTED_DDP.md
examples/13_ddp_simulation.py
```

## 14. Explicit non-goals

Frozen v0.10 does not implement:

- multiprocessing/multi-node transport;
- NCCL/Gloo/MPI;
- CUDA collectives;
- true communication/backward overlap;
- `no_sync()`;
- unused-parameter traversal;
- uneven-input join;
- FSDP/ZeRO;
- sharded/distributed checkpoint;
- production collective fault tolerance.

## 15. Next-version boundary

Graph capture/compiler concepts belong to `v0.11`, created from the latest validated `main`, not from this frozen branch.
