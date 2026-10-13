# Distributed / DDP Concepts — mini-pytorch v0.10

v0.10 is a deliberately **single-process simulation of replicated data parallelism**.
It does not pretend to provide NCCL, sockets, multiprocessing or GPU collectives.
The purpose is to make the architecture visible:

```text
data partitioning
      +
local backward
      +
gradient-ready events
      +
buckets
      +
collective reduction
      ↓
replica-consistent gradients
```

## 1. Why local training diverges

Assume two model replicas start from identical parameters `w`.

```text
rank 0                     rank 1
  │                          │
batch A                    batch B
  │                          │
backward                   backward
  │                          │
grad_A                     grad_B
```

If the two optimizers immediately update with different local gradients:

```text
w0' = w - lr * grad_A
w1' = w - lr * grad_B
```

then generally:

```text
w0' != w1'
```

Replicated data parallelism inserts a synchronization step:

```text
mean_grad = (grad_A + grad_B) / 2

w0' = w - lr * mean_grad
w1' = w - lr * mean_grad
```

Now the replicas remain equal.

For equal local batch sizes and a mean-reduced loss, this averaged gradient is also the gradient of the concatenated global batch.

---

## 2. v0.10 architecture

```text
                     global Dataset
                           │
                           ▼
                  DistributedSampler
                 ┌─────────┴─────────┐
                 ▼                   ▼
              rank 0              rank 1
              batch A             batch B
                 │                   │
                 ▼                   ▼
             Module 0            Module 1
                 │                   │
                 ▼                   ▼
          AutogradEngine 0     AutogradEngine 1
                 │                   │
                 ▼                   ▼
       leaf Parameter hooks  leaf Parameter hooks
                 │                   │
                 ▼                   ▼
              Reducer 0            Reducer 1
                 │                   │
            GradientBucket(s)  GradientBucket(s)
                 │                   │
                 └─────────┬─────────┘
                           ▼
                  ProcessGroup world
                           │
                       all_reduce
                           │
                           ▼
                   averaged buckets
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
           Parameter.grad      Parameter.grad
                 │                   │
                 ▼                   ▼
             optimizer           optimizer
```

Every layer has one job.

### `DistributedSampler`

Answers:

> Which global sample indices belong to this rank?

It never loads data and does not know about gradients.

### `ProcessGroup`

Answers:

> If all ranks participate in collective sequence `k`, what is the reduced value?

It does not know about Modules or optimizers.

### `Reducer`

Answers:

> Which parameter gradients are ready, how are they bucketed, flattened, reduced and written back?

It does not perform forward math.

### `DistributedDataParallel`

Answers:

> How does a Module get leaf-gradient hooks and a Reducer without changing its original forward/backward formulas?

---

## 3. Rank and world size

Create one rank-local ProcessGroup view per simulated replica:

```python
from mini_pytorch.distributed import create_process_group

groups = create_process_group(2)

print(groups[0].rank)       # 0
print(groups[1].rank)       # 1
print(groups[0].world_size) # 2
```

Required invariant:

```text
world_size >= 1
0 <= rank < world_size
```

All rank-local views share one `_InProcessWorld` object.

---

## 4. Why ProcessGroup returns Work

Real distributed collectives are not conceptually just ordinary pure functions. A rank can launch communication and observe completion later.

v0.10 models this with:

```python
work = group.all_reduce(array, op="mean")
```

The call **submits** this rank's contribution and returns `CollectiveWork`.

For two ranks:

```text
rank 0 submit sequence 0
      ↓
Work incomplete

rank 1 submit sequence 0
      ↓
collective complete
      ↓
both Work handles expose the same result
```

The simulator never blocks:

```python
w0 = groups[0].all_reduce([1.0])
w0.result()
```

fails while rank 1 has not submitted.

That is intentional. Sequential Python execution must not pretend it can block like two concurrently running worker processes.

---

## 5. Collective sequence ordering

Every rank-local ProcessGroup owns:

```text
next_sequence = 0, 1, 2, ...
```

The same sequence number across ranks must represent the same collective contract:

```text
operation
shape
dtype
```

For example this is invalid:

```text
rank 0 sequence 3: mean, shape=(8,), float64
rank 1 sequence 3: sum,  shape=(8,), float64
```

So v0.10 fails explicitly on:

- operation mismatch;
- shape mismatch;
- dtype mismatch;
- duplicate rank submission;
- reading an incomplete Work.

This introduces an important distributed-system idea:

> Collective operations are ordered protocols, not independent local calls.

---

## 6. DistributedSampler

For a dataset of 12 samples and 3 ranks with no shuffle:

```text
global order:
0 1 2 3 4 5 6 7 8 9 10 11

rank 0:
0 3 6 9

rank 1:
1 4 7 10

rank 2:
2 5 8 11
```

Implementation rule:

```text
create one global ordering
      ↓
indices[rank : total_size : num_replicas]
```

### `drop_last=False`

If dataset size is not divisible by world size, the sampler pads the global index list by repeating from the beginning until every rank gets the same number of samples.

### `drop_last=True`

The global ordering is truncated to a divisible size.

### `set_epoch`

```python
sampler.set_epoch(epoch)
```

changes shuffled order deterministically using:

```text
seed + epoch
```

This mirrors the important idea that every distributed rank must use the **same global shuffle** and then take a different partition.

---

## 7. DDP hooks do not change backward formulas

`DistributedDataParallel` registers a Tensor gradient hook on every trainable leaf Parameter.

The Autograd Engine already does:

```text
complete leaf gradient
      ↓
Tensor backward hooks
      ↓
persistent Parameter.grad
```

DDP inserts itself at the hook boundary:

```text
leaf gradient ready
      ↓
Reducer.mark_variable_ready(name, grad)
      ↓
return None
      ↓
Autograd continues normally
```

The hook only **stages** the rank-local gradient.

It does not all-reduce inside backward.

This preserves the architectural separation:

```text
Autograd
→ produces gradients

DDP Reducer
→ synchronizes gradients
```

---

## 8. Why v0.10 uses an explicit synchronization barrier

A real multi-process DDP implementation can launch asynchronous communication while backward is still running on other ranks.

Our CI simulation executes:

```text
rank 0 Python code
then rank 1 Python code
```

If rank 0's backward hook blocked waiting for rank 1, rank 1 would never run.

Therefore v0.10 uses two phases:

### Phase 1 — local backward

```text
rank 0 backward
→ local grads staged

rank 1 backward
→ local grads staged
```

### Phase 2 — explicit barrier

```python
synchronize_ddp([ddp0, ddp1])
```

Internally:

```text
all rank 0 buckets submit
all rank 1 buckets submit
      ↓
collectives complete
      ↓
all ranks read reduced Work results
      ↓
unflatten into Parameter.grad
```

Then and only then should the optimizer step.

---

## 9. Gradient buckets

Reducing every tiny parameter independently would create many collective calls.

DDP therefore groups gradients into buckets.

v0.10 uses a teaching limit:

```python
bucket_cap_numel=1024
```

Parameters are visited in **reverse registration order** because later layers often become gradient-ready earlier during reverse-mode autodiff.

A bucket records:

```text
bucket index
dtype
device
entry name
entry shape
entry numel
entry flat offset
```

Different dtype or device starts a new bucket.

Example:

```text
Linear.weight shape=(1,2) numel=2
Linear.bias   shape=(1,)  numel=1
bucket_cap_numel=2

reverse order:
bias → bucket 0
weight → bucket 1
```

---

## 10. Flatten → all-reduce → unflatten

Suppose a bucket contains:

```text
bias grad   shape=(2,)    numel=2
weight grad shape=(2,3)   numel=6
```

Reducer constructs:

```text
flat_grad shape=(8,)
```

The ProcessGroup reduces this flat buffer once.

After reduction:

```text
flat[0:2] → bias.grad.reshape(2)
flat[2:8] → weight.grad.reshape(2,3)
```

Offsets are recorded when the bucket is built, not guessed during restore.

---

## 11. Why the default reduction is mean

v0.10 defaults to:

```text
mean(local gradients across ranks)
```

With equal local batch sizes and a local loss using `reduction="mean"`:

```text
mean(rank-local mean gradients)
==
global concatenated-batch mean gradient
```

This makes the central correctness test straightforward.

If local batch sizes differ, naïve rank averaging is not generally the same as sample-weighted global-batch averaging. Uneven-input semantics are explicitly outside v0.10.

---

## 12. Central reference-equivalence proof

The test constructs identical models.

### Distributed path

```text
rank 0: batch[0:2] → backward
rank 1: batch[2:4] → backward
      ↓
synchronize_ddp
      ↓
SGD step on both replicas
```

### Reference path

```text
one model
batch[0:4]
mean MSE loss
      ↓
backward
      ↓
SGD step
```

CI verifies:

```text
replica 0 weight == replica 1 weight == reference weight
replica 0 bias   == replica 1 bias   == reference bias
```

This is the most important v0.10 invariant.

---

## 13. Initial module state

Replica synchronization only makes sense if replicas begin from the same parameters.

The helper:

```python
broadcast_module_state([ddp0, ddp1], src=0)
```

copies Parameter and Buffer values from one source replica to all others.

This is intentionally explicit in the in-process simulation. Production DDP systems typically synchronize initial module state through their distributed runtime.

---

## 14. Stale gradients and iteration boundaries

v0.10 intentionally supports the simple training lifecycle:

```text
optimizer.zero_grad()
      ↓
DDP forward
      ↓
backward
      ↓
synchronize_ddp
      ↓
optimizer.step()
```

If the next forward sees old `.grad` arrays that were not cleared, Reducer raises an error.

Why?

Because this version does not yet implement `no_sync()` or explicit multi-backward gradient accumulation semantics. Failing loudly is more educational than silently mixing accumulation policies.

---

## 15. Unused parameters

Every trainable Parameter known to the Reducer is expected to become ready.

If a model owns:

```text
used.weight
used.bias
unused
```

but `unused` does not participate in the graph, synchronization fails with the missing parameter name.

Real DDP has machinery for unused-parameter detection. v0.10 deliberately does not implement that traversal yet.

---

## 16. EDU backend

The collective simulator reduces host NumPy gradient values, while Parameter objects keep their existing Storage/device identity.

Tests verify two replicas placed on:

```text
edu
```

remain on `edu` after synchronization.

v0.10 does not claim real accelerator communication.

---

## 17. Source map

```text
mini_pytorch/distributed/
├── process_group.py
├── sampler.py
├── reducer.py
├── ddp.py
└── __init__.py
```

Recommended reading order:

```text
process_group.py
      ↓
sampler.py
      ↓
reducer.py
      ↓
ddp.py
      ↓
tests/test_distributed.py
      ↓
examples/13_ddp_simulation.py
```

---

## 18. What this maps to in real PyTorch thinking

The names intentionally transfer to real concepts:

```text
rank / world_size
ProcessGroup
collective sequence
Work
all_reduce
DistributedSampler
DistributedDataParallel
Reducer
gradient-ready hook
bucket
```

But the implementation is original, NumPy-first, synchronous at an explicit barrier, and single-process.

---

## 19. Explicit non-goals

Not implemented:

- real processes/nodes;
- TCP/shared-memory transport;
- Gloo/NCCL/MPI;
- CUDA streams/events;
- real overlap of backward and communication;
- `no_sync()`;
- unused-parameter graph detection;
- uneven-input join handling;
- FSDP/ZeRO;
- distributed checkpointing;
- fault-tolerant collectives.

Those require additional runtime concepts rather than more code inside this Reducer.
