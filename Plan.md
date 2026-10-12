# mini-pytorch — Architecture-Driven Development Plan

> Canonical roadmap for `main`.
>
> **Invariant:** `main` is always the newest integrated, validated state. Version branches are readable learning snapshots. A completed version is validated, frozen, then integrated into `main` before the next branch is created.

## 1. Project purpose

mini-pytorch removes one layer of PyTorch “magic” at a time with readable Python/NumPy implementations.

The roadmap follows architectural dependency order:

```text
Tensor / Autograd
      ↓
Module / Optimizer
      ↓
Data pipeline
      ↓
Tensor memory model
      ↓
Operator Dispatcher
      ↓
NN operator stack
      ↓
Device / Backend
      ↓
Advanced Autograd Engine
      ↓
Training Runtime / Reproducibility
      ↓
Distributed / DDP Concepts
      ↓
Graph IR / compiler stack
```

One version should introduce one dominant mental model and preserve earlier invariants.

---

# 2. Completed milestones

## v0.1 — Eager Tensor + Dynamic Autograd ✅

```text
Tensor → Function → dynamic graph → backward → Parameter → Module → Optimizer
```

Key mechanisms:

- Tensor / `requires_grad`;
- reverse-mode autodiff;
- dynamic graph;
- gradient accumulation;
- Module / Parameter / Linear / Sequential;
- SGD / Adam;
- numerical gradcheck.

## v0.2 — Optimizer State + LR Scheduler ✅

```text
Parameter
   ↓
Optimizer parameter groups + state
   ↓
LR Scheduler
```

Key mechanisms:

- enhanced SGD;
- Adam / AdamW / Adamax;
- RMSprop / Adagrad;
- parameter groups;
- optimizer state serialization;
- StepLR / MultiStepLR / ExponentialLR;
- CosineAnnealingLR / LinearLR / LambdaLR;
- ReduceLROnPlateau.

## v0.3 — Dataset / Sampler / DataLoader + MNIST ✅

```text
Dataset → Sampler → BatchSampler → DataLoader → Tensor batch
```

Key mechanisms:

- Dataset / TensorDataset;
- SequentialSampler / RandomSampler;
- BatchSampler;
- DataLoader;
- MNIST IDX reader;
- Flatten / CrossEntropyLoss;
- MLP MNIST training;
- Loguru / metrics / plots / checkpoints in examples.

## v0.4 — Tensor Memory Model ✅

```text
Storage + shape + stride + storage_offset + VersionCounter
```

Key mechanisms:

- explicit Storage;
- views / aliasing;
- transpose / permute / slicing;
- `view()` vs `reshape()`;
- contiguous semantics;
- inplace mutation;
- shared version counter;
- saved-version Autograd safety.

## v0.5 — Operator Registry + Dispatcher ✅

```text
Tensor API
   ↓
Operator Schema
   ↓
Dispatcher / DispatchKeySet
   ↓
Autograd / NoGrad / backend key
   ↓
redispatch
```

Key mechanisms:

- OperatorSchema;
- operator/kernel registry;
- DispatchKey / DispatchKeySet;
- dispatch priority;
- fallbacks;
- redispatch;
- backend numerical kernels;
- Autograd wrappers above backend kernels.

Validated baseline: **80 tests**.

## v0.6 — Neural-Network Operator Stack / CNN ✅

```text
nn.Module
    ↓
functional operator
    ↓
Dispatcher
    ↓
Autograd wrapper
    ↓ redispatch
backend NN kernel
```

Key mechanisms:

- Conv2d reference loops + im2col;
- Conv2d input/weight/bias gradients;
- MaxPool2d;
- Dropout;
- BatchNorm1d / BatchNorm2d;
- Parameter vs Buffer state;
- CNN MNIST example.

Validated baseline: **90 tests**.

## v0.7 — Device / Backend Abstraction ✅

```text
Tensor metadata
     ↓
Storage + Device
     ↓
backend-owned allocation
     ↓
Dispatcher
 ┌────┴────┐
 CPU      EDU
```

Key mechanisms:

- immutable Device;
- Backend abstraction/registry;
- CPU + deterministic educational `edu` backend;
- backend-owned Storage;
- Tensor.device / Tensor.to(device);
- cross-device alias break;
- differentiable transfer;
- device-aware DispatchKey inference;
- mixed-device errors;
- Module.to(device).

Validated baseline: **103 tests**.

## v0.8 — Advanced Autograd Engine ✅

```text
forward graph
     ↓
GradFn / Context
     ↓
AutogradEngine
 ┌────┼───────────────┐
 ▼    ▼               ▼
VJP  graph lifetime  hooks
 │    retain_graph    │
 ▼                    ▼
leaf grad       retained non-leaf grad
```

Key mechanisms:

- explicit AutogradEngine separated from Tensor;
- topological traversal and transient local gradients;
- Context release state;
- repeated-backward failure after release;
- `retain_graph=True`;
- non-leaf `retain_grad()`;
- gradient hooks + removable handles;
- custom Function teaching path;
- explicit higher-order-gradient boundary.

Validated baseline: **123 tests**.

## v0.9 — Training Runtime / Serialization / Reproducibility ✅

```text
Module / Optimizer / Scheduler
            +
       framework RNG
            +
       TrainingState
            ↓
      CheckpointState
            ↓
metadata.json + tensors.npz
            ↓
       fresh objects
            ↓
          restore
            ↓
deterministic continuation
```

Implemented:

- framework PCG64 RNG;
- `manual_seed / get_rng_state / set_rng_state`;
- framework RNG integration for Linear/Conv2d/default Dropout/default RandomSampler;
- explicit `TrainingState`;
- schema-versioned `CheckpointState`;
- inspectable JSON + NPZ serialization;
- nested optimizer state encoding without pickle;
- staged/rename checkpoint publication;
- model/optimizer/scheduler restoration;
- RNG restore-last semantics;
- target CPU/EDU device preservation;
- strict schema/shape/group/scheduler failures;
- deterministic uninterrupted-vs-resumed integration proof.

Central proof compares exact:

```text
loss trajectory
model state
Adam moments/counters
StepLR state/current LR
TrainingState
future framework RNG draws
```

Validated baseline: **140 tests**.

---

# 3. Current next milestone — v0.10 Distributed / DDP Concepts 🎯

## 3.1 Why distributed comes now

By v0.9, a single training process has explicit mechanisms for:

```text
Tensor execution
Autograd gradient generation
Tensor gradient hooks
Optimizer state
Data sampling
Device/backend placement
Training runtime/checkpoint state
```

The next architectural question is no longer how one process trains, but how multiple replicas cooperate.

Suppose two replicas start from identical parameters:

```text
Replica rank 0              Replica rank 1
      │                            │
 batch A                        batch B
      │                            │
 forward                        forward
      │                            │
 backward                       backward
      │                            │
 grad_A                         grad_B
```

If each optimizer immediately uses its own local gradient:

```text
parameter_0' != parameter_1'
```

The replicas diverge.

DDP's core idea is conceptually:

```text
local gradients
      ↓
collective reduction
      ↓
shared averaged gradients
      ↓
identical optimizer updates
```

The central v0.10 question is:

> How should dataset partitioning, rank/world-size semantics, gradient-ready events, gradient buckets and collective reduction compose so multiple replicas remain equivalent after every optimizer step?

---

## 3.2 Target architecture

```text
                 global Dataset
                      │
                      ▼
              DistributedSampler
              ┌───────┴────────┐
              ▼                ▼
           rank 0            rank 1
           indices           indices
              │                │
              ▼                ▼
          replica 0         replica 1
              │                │
          local loss        local loss
              │                │
              ▼                ▼
       AutogradEngine     AutogradEngine
              │                │
              ▼                ▼
      parameter hooks    parameter hooks
              │                │
              └──────┬─────────┘
                     ▼
                 DDP Reducer
                     │
             gradient buckets
                     │
                     ▼
                ProcessGroup
                     │
                  all_reduce
                     │
                     ▼
             averaged gradients
                     │
                     ▼
             Optimizer.step()
                     │
                     ▼
          replica parameters equal
```

Keep the following responsibilities separate:

```text
DistributedSampler
→ which data indices belong to a rank

Reducer
→ which parameter gradients are ready / grouped

ProcessGroup
→ how values from ranks are collectively combined

Optimizer
→ how synchronized gradients update parameters
```

---

## 3.3 v0.10.1 — Distributed context: rank and world_size

Introduce a small explicit distributed context/value object.

Possible API:

```python
DistributedContext(
    rank=0,
    world_size=2,
)
```

Required validation:

```text
world_size >= 1
0 <= rank < world_size
```

The object should make rank/world-size explicit rather than relying on hidden globals or environment variables for the educational version.

Concepts to document:

```text
world_size
→ number of participating replicas

rank
→ this replica's unique position in [0, world_size)
```

Tests:

- valid contexts;
- invalid negative rank;
- rank >= world_size;
- invalid world_size.

---

## 3.4 v0.10.2 — ProcessGroup and collective semantics

Introduce an abstract ProcessGroup boundary.

Conceptual interface:

```python
class ProcessGroup:
    def all_reduce(...):
        ...
```

Do **not** tie DDP directly to NumPy global helpers or network code.

For v0.10 use a deterministic in-process implementation, e.g.:

```text
InProcessCollectiveWorld
        │
        ├── contribution from rank 0
        ├── contribution from rank 1
        └── ...
        ↓
ProcessGroup collective round
        ↓
sum / mean result
```

### Collective invariants

For one logical all-reduce round, all ranks must agree on:

- world size;
- collective sequence/round id;
- tensor shape;
- dtype;
- reduction operation.

At minimum support:

```text
SUM
MEAN
```

DDP gradient synchronization should normally use mean semantics so it can be compared to a reference global-batch gradient.

### Important simulation constraint

Ordinary CI executes rank simulations sequentially. A hook on rank 0 cannot synchronously block waiting for rank 1, because rank 1 backward has not executed yet.

Therefore v0.10 should not model network collectives as blocking calls from inside gradient hooks.

Use an educational two-phase structure:

```text
phase 1: local backward
         ↓
         hooks stage / mark local grads ready

phase 2: explicit synchronization barrier
         ↓
         collective combines all replicas
         ↓
         synchronized grads written back
```

This keeps the dependency graph correct without inventing fake concurrency.

---

## 3.5 v0.10.3 — DistributedSampler

Add a rank-aware sampler inspired by `torch.utils.data.DistributedSampler`.

Possible API:

```python
DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=0,
    drop_last=False,
)

sampler.set_epoch(epoch)
```

Core rule:

```text
same seed + same epoch
→ all ranks derive the same global permutation
→ each rank selects a deterministic strided partition
```

For example:

```text
global permutation:
[7, 2, 5, 1, 6, 0, 3, 4]

rank 0 / world_size 2:
[7, 5, 6, 3]

rank 1 / world_size 2:
[2, 1, 0, 4]
```

Need explicit behavior when dataset size is not divisible by world size:

```text
drop_last=True
→ truncate globally to evenly divisible size

drop_last=False
→ deterministically pad/repeat indices to equal replica lengths
```

Why equal replica lengths matter:

```text
same number of local training iterations
→ same number/order of collectives
```

`set_epoch(epoch)` must change shuffle deterministically while keeping all ranks aligned.

Tests:

- partitions cover expected global indices;
- ranks do not overlap when divisible and no padding is required;
- deterministic padding behavior;
- `drop_last` behavior;
- same seed+epoch reproducibility;
- different epoch changes permutation;
- invalid rank/world-size errors.

---

## 3.6 v0.10.4 — DDP wrapper boundary

Introduce an educational wrapper, e.g.:

```python
DistributedDataParallel(
    module,
    process_group=...,
    bucket_cap=...,
)
```

The wrapper should preserve normal Module usage:

```python
output = ddp(input)
```

and delegate forward directly to the wrapped module.

DDP should not implement different forward math. Its main responsibility is gradient synchronization after backward.

Possible structure:

```text
DistributedDataParallel
├── module
├── process_group / context
└── Reducer
```

Parameters stay normal Parameter objects. Optimizers may continue to receive:

```python
optimizer = SGD(ddp.parameters(), ...)
```

or the underlying module parameters, as long as the behavior is documented and tested.

---

## 3.7 v0.10.5 — Gradient-ready hooks and Reducer

v0.8 already added Tensor hooks. v0.10 should reuse them instead of changing Autograd formulas.

For each trainable parameter:

```text
backward computes local parameter grad
       ↓
parameter hook fires
       ↓
Reducer marks parameter ready
       ↓
local grad copied/staged for bucket
```

The hook should **not** perform a blocking all-reduce in the single-process simulation.

Reducer responsibilities:

- register/remove parameter hooks;
- map Parameter → bucket location;
- record ready state;
- reject duplicate-ready events in one iteration when inappropriate;
- expose whether all expected local grads are ready;
- reset iteration state after synchronization.

Important invariant:

```text
Reducer observes Autograd
but does not become AutogradEngine
```

Tests:

- hooks fire when parameters receive grad;
- parameters with no gradient are handled according to documented policy;
- duplicate/incorrect readiness errors are clear;
- iteration reset works.

---

## 3.8 v0.10.6 — Gradient buckets

Real DDP groups many gradients into buckets to reduce communication overhead.

Educational v0.10 should model the mechanism even though in-process communication is cheap.

Example:

```text
Parameter A grad shape (4, 8)
Parameter B grad shape (4,)
Parameter C grad shape (2, 4)
          │
          ▼
flatten in deterministic parameter order
          │
          ▼
Bucket 0 [ A | B ]
Bucket 1 [ C ]
```

Required metadata per bucket:

- parameter order;
- flat offsets;
- original shapes;
- dtype;
- total element count;
- ready count/state.

Synchronization:

```text
local bucket rank 0
local bucket rank 1
       ↓
ProcessGroup all_reduce(mean)
       ↓
reduced flat bucket
       ↓
unflatten
       ↓
write synchronized arrays into Parameter.grad
```

Bucket construction should be deterministic across replicas with identical model structure.

Do not add compression, fp16 communication or overlap optimization yet.

---

## 3.9 v0.10.7 — Synchronization barrier API

Because ranks are simulated sequentially, expose a clear synchronization phase rather than hiding it.

Possible educational API forms:

```python
synchronize_ddp([ddp_rank0, ddp_rank1])
```

or:

```python
world.synchronize(reducers)
```

The API should:

1. verify all ranks are represented exactly once;
2. verify reducers have compatible bucket layout;
3. verify local gradients are ready;
4. reduce matching buckets across ranks;
5. divide by world size for mean semantics;
6. write grads back to each replica;
7. reset synchronization state for the next iteration.

This explicit barrier corresponds conceptually to the synchronization that a real asynchronous ProcessGroup would coordinate during backward.

---

## 3.10 v0.10.8 — Replica equivalence integration test

This is the central v0.10 proof.

### Setup

Create one reference model and `world_size=2` replica models with identical initial state.

Global batch:

```text
[x0, x1, x2, x3]
```

Reference path:

```text
one model
  ↓
full batch of 4
  ↓
mean loss
  ↓
backward
  ↓
global-batch gradient
  ↓
optimizer.step
```

Distributed simulation:

```text
rank 0 sees first local batch
rank 1 sees second local batch
          ↓
local backward independently
          ↓
DDP synchronize(mean)
          ↓
each rank receives average local gradient
          ↓
optimizer.step independently
```

With equally sized local batches and mean loss, the synchronized gradient should match the reference global-batch gradient.

Assertions:

```text
rank0 synchronized grad == reference grad
rank1 synchronized grad == reference grad

rank0 params after step == reference params
rank1 params after step == reference params
rank0 params == rank1 params
```

Run multiple steps, not just one, to prove replicas remain locked together.

Add a DistributedSampler variant of the experiment to show data partitioning and synchronization compose correctly.

---

## 3.11 v0.10.9 — Collective and DDP failure semantics

Tests should explicitly cover errors such as:

- invalid rank/world size;
- duplicate rank registration;
- missing rank contribution;
- mismatched tensor shapes;
- mismatched dtype;
- incompatible bucket count/order/layout;
- synchronization before local backward is ready;
- synchronization twice without a new backward iteration;
- replica model parameter layout mismatch;
- invalid reduction operation.

Do not silently skip a rank or parameter to make synchronization complete.

---

## 3.12 Suggested source layout

```text
mini_pytorch/distributed/
├── __init__.py
├── context.py
├── process_group.py
├── sampler.py
├── reducer.py
└── ddp.py

tests/test_distributed.py
docs/DISTRIBUTED_DDP.md
examples/13_ddp_simulation.py
```

Keep responsibilities narrow:

```text
context.py
→ rank/world_size values

process_group.py
→ collective semantics/coordinator

sampler.py
→ dataset partitioning

reducer.py
→ grad readiness/bucket layout/writeback

ddp.py
→ Module-facing wrapper + synchronization integration
```

---

## 3.13 v0.10 educational example

Add:

```text
examples/13_ddp_simulation.py
```

It should demonstrate:

```text
build reference model
      ↓
clone two replica states
      ↓
construct DistributedSampler / contexts
      ↓
local backward on each rank
      ↓
print local gradients
      ↓
DDP synchronize
      ↓
print averaged gradients
      ↓
optimizer.step
      ↓
prove replicas and reference match
```

Use Loguru for the walkthrough, keep data tiny, and require no multiprocessing/GPU/network.

---

## 3.14 v0.10 documentation

Add:

```text
docs/DISTRIBUTED_DDP.md
```

It should explain:

- process vs rank vs world_size;
- data parallel vs model parallel distinction;
- why DistributedSampler exists;
- local gradient vs synchronized gradient;
- all-reduce sum vs mean;
- ProcessGroup as communication abstraction;
- why DDP uses Autograd hooks;
- why gradients are bucketed;
- why identical initial model state is required;
- why collective ordering must match across ranks;
- why the educational simulation uses an explicit barrier instead of blocking hook calls;
- mapping from the simulation to real PyTorch DDP/NCCL architecture.

Version branch must update:

```text
README.md
version.md
Agents.md
```

---

## 3.15 Definition of Done for v0.10

Before freeze, tests must demonstrate:

- explicit rank/world-size validation;
- deterministic DistributedSampler partitioning;
- epoch-dependent shared shuffle;
- ProcessGroup abstraction exists independently from DDP;
- simulated SUM/MEAN collective correctness;
- clear collective mismatch failures;
- DDP wrapper delegates normal forward;
- Tensor hooks mark local parameter gradients ready;
- deterministic bucket construction;
- flatten/reduce/unflatten correctness;
- synchronized grads written back before optimizer step;
- all replicas remain parameter-identical across multiple steps;
- synchronized DDP result matches a single-model global-batch reference;
- v0.9 deterministic checkpoint-resume regression still passes;
- all earlier regression tests remain green;
- architecture docs/example exist;
- exact final v0.10 branch-head GitHub Actions passes;
- frozen tree is integrated into `main`.

### Explicit non-goals for v0.10

Do not add yet:

- multiprocessing launchers;
- TCP/Gloo/NCCL transport;
- GPU communication;
- asynchronous communication overlap;
- gradient compression;
- fp16/bf16 communication hooks;
- FSDP / ZeRO / sharded parameters;
- distributed checkpointing;
- RPC;
- distributed Autograd across arbitrary graphs.

The goal is architectural understanding of DDP's synchronization path.

---

# 4. Future milestones

## v0.11 — Graph IR and Eager-to-Graph Capture

Core question:

> How can eager operator execution be represented explicitly as a graph of values and nodes without changing numerical results?

Planned concepts:

- Node / Value / Graph;
- supported-op tracing;
- graph interpreter;
- explicit graph breaks;
- eager-vs-graph equivalence;
- graph visualization.

## v0.12 — Dynamo-like Guards and Graph Cache

Core concepts:

- shape/dtype/device/constant guards;
- graph cache;
- guard failure;
- recompilation;
- basic dynamic-shape experiments.

## v0.13 — AOTAutograd-like Forward/Backward Capture

Core concepts:

- captured forward graph;
- captured backward graph;
- saved-intermediate contract;
- eager vs captured gradient equivalence;
- foundations for differentiable/lowered backward graphs.

## v0.14 — Inductor-like Lowering and Kernel Fusion

Core concepts:

- lower-level loop IR;
- pointwise fusion;
- reduction lowering;
- generated Python/NumPy kernels;
- eager vs compiled correctness/performance comparison.

## v0.15 — Runtime / Performance Mechanics

Potential concepts after compiler architecture is understandable:

- allocator/caching concepts;
- scheduling;
- prefetching;
- kernel launch/fusion accounting;
- profiling hooks.

## v1.0 — Integrated Architecture

Target educational map:

```text
Tensor / Storage
      ↓
Dispatcher / Backend
      ↓
AutogradEngine
      ↓
Module / Optimizer / Data
      ↓
Training Runtime
      ↓
Distributed Reducer
      ↓
Graph Capture / Guards
      ↓
AOT backward
      ↓
Lowering / Fusion
```

The project reaches v1.0 when these layers compose cleanly while historical version branches remain readable snapshots of how the architecture grew.
