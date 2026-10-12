# mini-pytorch — Architecture-Driven Development Plan

> Canonical roadmap for `main`.
>
> **Invariant:** `main` is always the newest integrated project state. Version branches are readable learning snapshots. A completed version is validated, frozen, then integrated into `main` before the next version branch is created.

---

# 1. Project purpose

mini-pytorch is not trying to reproduce the entire PyTorch API. It exists to remove one layer of framework “magic” at a time with small, readable Python/NumPy implementations.

The long-term learning path should answer:

```text
What is a Tensor?
      ↓
How does Autograd build and execute a graph?
      ↓
How are parameters/modules/optimizers organized?
      ↓
How does data become batches?
      ↓
How do Storage, stride and views work?
      ↓
Why does PyTorch need an operator Dispatcher?
      ↓
How can multiple backends implement one operator API?
      ↓
How do advanced Autograd/runtime/distributed mechanisms work?
      ↓
How does eager Python execution become compiled graphs/kernels?
```

The project follows **architectural dependency order**, not feature popularity.

---

# 2. Reference architecture

```text
Python Tensor / Module API
            │
            ▼
    Tensor memory model
Storage + shape + stride + offset
            │
            ▼
      Operator system
 schema + registry + Dispatcher
            │
      ┌─────┴───────────┐
      ▼                 ▼
 backend kernels      Autograd
 CPU / future accel   wrappers/nodes
      │                 │
      └──────┬──────────┘
             ▼
        NN operators
             │
             ▼
      Optimizer / LR

Dataset → Sampler → DataLoader
             │
             ▼
       training runtime
             │
      ┌──────┴───────────┐
      ▼                  ▼
 distributed          compiler
 DDP-like             graph capture
 collectives          lowering/fusion
```

mini-pytorch reproduces these **conceptual boundaries** with original educational code. Exact PyTorch source structure and edge-case parity are not goals.

---

# 3. Repository/version rules

## `main`

`main` is the newest integrated state and contains:

- canonical `Plan.md`;
- canonical `Agents.md`;
- latest integrated `version.md`;
- latest validated implementation.

## Version branches

Every `v0.x` / `v1.x` branch must contain:

- `version.md` — identity, delta, validation and next boundary;
- `Agents.md` — branch-specific scope contract;
- tests for the new mechanism;
- architecture documentation for new subsystems;
- an educational example when appropriate.

## Version lifecycle

```text
latest main
    ↓
create v0.x
    ↓
write version.md + branch Agents.md
    ↓
implement one major mechanism
    ↓
tests + error paths + integration example
    ↓
exact remote CI
    ↓
freeze version.md + Agents.md
    ↓
synchronize to main
    ↓
advance Plan.md
    ↓
create next version from newest main
```

---

# 4. Completed versions

## v0.1 — Eager Tensor + dynamic Autograd MVP ✅

### Learning question

How can a small framework dynamically build a computation graph during forward execution and automatically propagate gradients backward?

### Core mechanisms

- Tensor / requires_grad / leaf grad accumulation
- `Function`, `Context`, `GradFn`
- reverse topological backward traversal
- broadcasting-aware gradients
- Parameter / Module registration
- Linear / activations / Sequential
- MSELoss
- SGD / Adam
- state_dict basics

### Mental model

```text
Tensor operation
      ↓
Function.forward
      ↓
GradFn node
      ↓
dynamic DAG
      ↓
reverse traversal
      ↓
Function.backward
```

---

## v0.2 — Optimizer state + LR scheduling ✅

### Learning question

Why are model parameters, gradients, optimizer state and learning-rate state separate objects with different lifetimes?

### Core mechanisms

- Optimizer base + parameter groups
- per-parameter optimizer state
- SGD momentum / dampening / Nesterov
- Adam / AdamW / Adamax / RMSprop / Adagrad
- optimizer checkpoint state
- scheduler base
- StepLR / MultiStepLR / ExponentialLR
- CosineAnnealingLR / LinearLR / LambdaLR
- ReduceLROnPlateau

### Mental model

```text
Parameter + grad
      ↓
Optimizer state
      ↓
parameter update
      ▲
      │
Scheduler mutates group LR
```

---

## v0.3 — Data pipeline + MNIST classification ✅

### Learning question

Why does a framework separate data storage, index selection, batching and collation?

### Core mechanisms

```text
Dataset
   ↓
Sampler
   ↓
BatchSampler
   ↓
DataLoader
   ↓
collate_fn
   ↓
Tensor batch
```

Implemented:

- Dataset / TensorDataset
- SequentialSampler / RandomSampler
- BatchSampler
- sampler-driven DataLoader
- MNIST IDX parsing/downloading
- Flatten
- CrossEntropyLoss
- MLP MNIST training path
- Loguru logging
- metrics CSV
- loss/accuracy plotting
- model checkpoints

### Deferred extensions

The following are useful data-ecosystem extensions but are **not prerequisites for the architectural sequence** and therefore do not block v0.4+:

- Subset / random_split
- SubsetRandomSampler / WeightedRandomSampler
- explicit DataLoader iterator/fetcher classes
- confusion matrix / prediction grid
- richer resume-training utility
- multiprocessing/prefetch/pinned-memory work

These may be added later when their mechanism becomes a learning target.

---

## v0.4 — Tensor memory model: Storage / stride / views ✅

### Learning question

How can multiple Tensor objects expose different shapes/layouts while sharing one allocation, and why does aliasing matter for Autograd?

### Completed architecture

```text
Storage
├── flat allocation
└── shared version counter
       ▲
       │
 ┌─────┴──────────┐
 │                │
Tensor A        Tensor B
shape           shape
stride          stride
offset          offset
```

### Core mechanisms

- explicit `Storage`
- shape / stride / `storage_offset`
- contiguous-stride generation
- `is_contiguous()`
- transpose / permute metadata views
- basic slicing views, including negative stride
- `view()` aliases-or-fails semantics
- `reshape()` alias-or-copy semantics
- `contiguous()` materialization
- tracked `add_` / `mul_` / `copy_`
- shared Storage version counter
- saved-value version checks before backward
- optimizer/state-loading version integration
- correct 0-D scalar Tensor preservation

### Validation

Exact remote v0.4 implementation passed:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
65 / 65 tests
```

### Key document

```text
docs/TENSOR_MEMORY.md
```

---

# 5. Current next version

# v0.5 — Operator schema, registry and Dispatcher 🎯 NEXT

## Why v0.5 must follow v0.4

v0.4 established what a Tensor *is*. The next architectural problem is how Tensor operations are routed.

Without a dispatcher, adding devices/backends tends to produce this anti-pattern:

```python
if tensor.device == "cpu":
    ...
elif tensor.device == "cuda":
    ...
if requires_grad:
    ...
```

spread throughout Tensor methods.

A dispatcher creates one routing boundary:

```text
Tensor.add(other)
      ↓
Operator schema
      ↓
Dispatcher
      ↓
DispatchKeySet
  ┌───┼──────────────┐
  ▼   ▼              ▼
CPU  Autograd    future accelerator
kernel wrapper         kernel
```

## v0.5.1 Operator schema

Implement:

- `OperatorName` / namespace + name
- `OperatorSchema`
- simplified argument/return metadata
- schema registry
- duplicate-schema error
- lookup/introspection API

Target examples:

```text
mini::add.Tensor
mini::mul.Tensor
mini::matmul
mini::relu
mini::sum
```

## v0.5.2 Dispatch keys

Educational initial keys:

- `CPU`
- `Autograd`
- `NoGrad` or an equivalent grad-mode routing concept
- reserved accelerator key for later work

Implement:

- `DispatchKey`
- `DispatchKeySet`
- key priority
- key removal for redispatch

## v0.5.3 Kernel registry

Implement:

- register kernel for `(operator, dispatch_key)`
- query dispatch table
- duplicate registration policy
- missing-kernel error
- optional fallback registration

## v0.5.4 Dispatcher

Implement:

```text
call operator
    ↓
compute key set
    ↓
select highest-priority implementation
    ↓
invoke kernel/wrapper
    ↓
optional redispatch with key removed
```

The redispatch concept is important because an Autograd wrapper should be able to do graph work and then call the lower backend implementation.

## v0.5.5 Integrate a small operator subset

Migrate only enough operators to prove the boundary:

- add
- mul
- matmul
- relu
- sum

Do not migrate every op immediately.

## v0.5.6 Autograd routing demonstration

Demonstrate conceptually:

```text
add call
  ↓
Autograd wrapper
  ├── construct gradient metadata/node
  ↓ redispatch
CPU add kernel
```

This may coexist temporarily with the existing `Function` machinery; the goal is to teach routing, not to rewrite all Autograd in one version.

## v0.5 required tests

- schema registration/lookup
- duplicate registration
- dispatch-key priority
- missing implementation error
- CPU kernel selection
- Autograd wrapper selection
- redispatch reaches lower kernel
- no-grad path bypasses graph wrapper where intended
- migrated ops retain current numerical results
- migrated ops retain backward correctness
- Storage/view/version invariants remain green

## v0.5 Definition of Done

```text
[ ] operator schemas are explicit
[ ] registry is inspectable
[ ] dispatch keys have deterministic priority
[ ] kernels register by operator + key
[ ] dispatcher routes selected operators
[ ] redispatch works
[ ] Autograd/backend separation is demonstrable
[ ] previous 65-test baseline remains green
[ ] new dispatcher tests are green
[ ] docs/DISPATCHER.md exists
[ ] educational dispatcher example exists
[ ] version.md + Agents.md completed
[ ] exact remote CI passes
[ ] validated snapshot synchronized to main
```

### Explicit v0.5 non-goals

- Conv2d
- real CUDA
- device transfer
- DDP
- compiler graph capture
- full PyTorch dispatch-key set
- exact ATen schema parser

---

# 6. Future versions

## v0.6 — NN operator stack + CNN

### Why after Dispatcher

Conv2d becomes the first substantial operator proving that nontrivial kernels can live behind the routing system instead of being special-cased in Tensor.

### Planned mechanisms

- Conv2d readable loop reference
- Conv2d backward for input/weight/bias
- im2col educational optimization
- MaxPool2d
- Dropout
- BatchNorm1d / BatchNorm2d
- running-stat buffers
- CNN MNIST example

### Learning questions

- how parameter sharing changes gradients;
- why im2col converts convolution to GEMM-like work;
- parameters vs buffers;
- train/eval-dependent behavior.

---

## v0.7 — Device + backend abstraction

### Goal

Separate Tensor/Dispatcher semantics from the numerical backend.

```text
Tensor
  ↓
Dispatcher
  ├── NumPy CPU backend
  └── optional accelerator backend
```

Planned:

- `Device`
- device metadata on Storage/Tensor
- `.to(device)`
- backend-owned allocation
- device mismatch errors
- CPU kernel registrations
- optional CuPy experiment if practical

Non-goal: production CUDA C++ runtime.

---

## v0.8 — Advanced Autograd engine

Planned:

- leaf/non-leaf grad semantics refinement
- `retain_grad()`
- hooks
- graph lifetime/release
- `retain_graph`
- custom Function example
- richer saved-tensor handling
- optional `create_graph`
- second-order derivative demonstration

Learning target: understand not only differentiation formulas but graph lifetime and ownership.

---

## v0.9 — Training runtime, checkpointing and reproducibility

Planned state:

```text
model state
optimizer state
scheduler state
RNG state
epoch / global_step
sampler state where relevant
```

Goals:

- unified checkpoint object
- deterministic save/resume
- resume MNIST/CNN without silently changing training trajectory
- seed lifecycle documentation

---

## v0.10 — Distributed / DDP concepts

Staged plan:

1. DistributedSampler
2. ProcessGroup abstraction
3. simulated/CPU `all_reduce`
4. DDP wrapper
5. gradient-ready hooks
6. bucket concept

Goal: explain model replicas + collective gradient synchronization, not reimplement NCCL.

---

## v0.11 — Graph IR + eager-to-graph capture

Planned:

- Node / Value / Graph
- constants and graph inputs/outputs
- trace supported operators
- graph pretty printer
- graph interpreter
- explicit graph-break concept

```text
Python eager code
      ↓
supported region
      ↓
Graph IR
      ↓
unsupported Python
      ↓
graph break / eager resume
```

---

## v0.12 — Dynamo-like guards + graph cache

Planned:

- shape/dtype/device/constant guards
- graph cache
- cache hits/misses
- guard failure → recompile
- dynamic-shape discussion and limited experiment

Learning target: captured graphs are valid only under recorded assumptions.

---

## v0.13 — AOTAutograd-like forward/backward graphs

Planned:

- forward graph capture
- explicit saved intermediates
- backward graph for supported operator subset
- eager vs captured gradient parity
- joint forward/backward representation

Learning target: backward computation can itself become compiler-visible work.

---

## v0.14 — Inductor-like lowering + fusion

Planned educational pipeline:

```text
Graph IR
   ↓
normalized/lowered ops
   ↓
loop-level representation
   ↓
fusion groups
   ↓
NumPy/Python generated kernel or pseudo-codegen
```

Focus:

- operator fusion
- intermediate elimination
- simple scheduling decisions
- compare unfused vs fused execution

Do not attempt production Triton/LLVM code generation in the first implementation.

---

## v0.15 — Runtime/performance mechanisms

Potential learning topics after the architecture is stable:

- allocator/caching allocator model
- DataLoader worker/prefetch lifecycle
- pinned-memory concept
- async stream/event simulation
- profiling
- kernel launch/overhead discussion
- memory planning

Only add mechanisms that expose a useful PyTorch architectural idea.

---

# 7. v1.0 target

v1.0 is not defined by feature count. It is reached when the framework provides a coherent educational map across the major layers:

```text
Tensor memory model
      ↓
Dispatcher/operator system
      ↓
backend kernels + Autograd
      ↓
NN + optim + data
      ↓
training/checkpoint runtime
      ↓
distributed concepts
      ↓
graph capture + compiler concepts
```

The user should be able to move from a mini-pytorch implementation to real PyTorch internals and recognize the same architectural boundaries.

---

# 8. Cross-version engineering rules

## Rule 1 — One major mental model per version

Avoid mixing Dispatcher + Conv2d + Device + DDP in one version.

## Rule 2 — Preserve earlier invariants

Later versions must retain earlier guarantees, especially:

- correct Autograd accumulation;
- optimizer state behavior;
- data-pipeline separation;
- Storage aliasing;
- scalar shape `()` semantics;
- view/copy distinction;
- version-counter correctness.

## Rule 3 — Reference implementation before optimization

```text
readable
  ↓
correct
  ↓
tested
  ↓
profiled
  ↓
optimized
```

## Rule 4 — Every abstraction needs error tests

A class existing is not enough. Its invariants and failure modes must be observable.

## Rule 5 — `main` never intentionally falls behind

Validated desired work is integrated immediately.

## Rule 6 — Every frozen branch explains itself

Opening any version branch should be enough to understand it via:

```text
Agents.md
version.md
architecture docs
example
source
Tests
```

---

# 9. Immediate next action

The next feature branch should be:

```text
v0.5
```

created from the latest `main`.

Its first commits should establish:

```text
version.md
Agents.md
mini_pytorch/dispatch/
    schema.py
    keys.py
    registry.py
    dispatcher.py
tests/test_dispatcher.py
docs/DISPATCHER.md
```

Only after that boundary is stable should existing Tensor operators be incrementally migrated through it.