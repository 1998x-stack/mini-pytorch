# mini-pytorch — Architecture-Driven Development Plan

> This document is the canonical development plan for `main`.
>
> **Repository rule:** `main` must always represent the newest integrated, validated state. Historical version branches (`v0.x`) are learning snapshots. Active development may occur on a version branch, but once validated it must be synchronized back into `main` immediately so that `main` never lags behind the newest working implementation.

---

# 1. Why this project exists

The purpose of mini-pytorch is not to recreate every PyTorch API. The purpose is to remove one layer of “magic” at a time by implementing the core mechanisms in readable Python/NumPy.

The project should answer questions such as:

- what is a Tensor beyond an ndarray?
- how is a dynamic computation graph constructed?
- why can `backward()` traverse a graph automatically?
- why do Parameters belong to Modules?
- why does an Optimizer own state separately from the model?
- why is data sampling separate from data storage?
- what are Storage, stride, offset, view, reshape and contiguous?
- why does PyTorch need a Dispatcher?
- how can the same operator dispatch to CPU, CUDA, Autograd or other backends?
- why does inplace mutation interact with autograd version counters?
- how does DDP synchronize gradients?
- how does `torch.compile` move from eager Python execution to captured graphs and generated kernels?

The desired end state is a compact educational framework whose architecture can be mapped subsystem-by-subsystem to real PyTorch.

---

# 2. Architectural reference model

The roadmap follows the dependency structure of real PyTorch rather than simply adding popular neural-network features.

```text
Python-facing Tensor / Module API
            │
            ▼
     Tensor implementation
 Storage + size + stride + offset
            │
            ▼
       Operator system
 schema / registry / dispatcher
            │
      ┌─────┴───────────┐
      ▼                 ▼
 backend kernels      Autograd
 CPU / GPU-like       graph nodes
      │                 │
      └──────┬──────────┘
             ▼
           nn
      layers / losses
             │
             ▼
        optim / lr
             │
             ▼
     Dataset / DataLoader
             │
             ▼
       training runtime
             │
      ┌──────┴──────────┐
      ▼                 ▼
 distributed         compiler
 DDP-like            graph capture
 collectives         lowering/codegen
```

Real PyTorch’s low-level Tensor representation separates storage from view metadata such as sizes, strides and storage offset; this is what makes multiple Tensor views able to alias the same underlying data. PyTorch also uses an operator dispatcher so one logical operator can select different kernels and behavior according to dispatch keys and backend/autograd state. The compiler stack later captures eager execution into graphs and lowers them through compiler backends.

mini-pytorch should reproduce these **conceptual boundaries**, not copy PyTorch source code or every edge case.

---

# 3. Branch and version model

## 3.1 `main`

`main` is always the newest integrated state.

Required invariants:

1. `main` must never intentionally lag behind an active version branch.
2. only validated code should be integrated into `main`;
3. `main/Plan.md` is the canonical forward roadmap;
4. `main/Agents.md` is the canonical engineering contract;
5. all version branches must contain `version.md` and `Agents.md`;
6. historical version branches should not receive future-version features.

## 3.2 Version branches

Version branches are educational snapshots.

```text
main  ─────────────────────────────────────────────► newest integrated state
  │
  ├──── v0.1   eager Tensor/autograd MVP
  │
  ├────────── v0.2   optimizer/scheduler
  │
  ├──────────────── v0.3   data pipeline/MNIST
  │
  ├────────────────────── v0.4   Tensor memory model
  │
  └──────────────────────────── ...
```

Every version branch must include:

- `version.md` — what this version means and how it differs from the previous version;
- `Agents.md` — development constraints for any future agent working on that branch;
- tests for the mechanisms introduced by the version;
- at least one educational example when appropriate.

## 3.3 Version completion rule

A version is not complete merely because its classes exist.

It is complete only when:

```text
mechanism implemented
       ↓
unit tests
       ↓
error-path tests
       ↓
integration example
       ↓
documentation
       ↓
version.md updated
       ↓
Agents.md constraints satisfied
       ↓
validated state synchronized to main
```

---

# 4. Completed / current versions

# v0.1 — Eager Tensor + dynamic Autograd MVP

## Core idea

Build the smallest complete training path:

```text
Tensor
  ↓
operator
  ↓
dynamic graph
  ↓
backward
  ↓
Parameter
  ↓
Module
  ↓
Loss
  ↓
Optimizer
```

## Mechanisms

- Tensor wrapper
- `requires_grad`
- gradient accumulation
- dynamic computation graph
- `Function` / backward context
- reverse-mode autodiff
- common differentiable operators
- Parameter
- Module
- Linear
- activations
- Sequential
- MSELoss
- SGD
- Adam
- state dictionaries

## What v0.1 intentionally hides

- explicit Storage/stride semantics
- dispatcher/backend selection
- production-grade DataLoader
- CUDA
- multiprocessing
- distributed training
- compiler stack

---

# v0.2 — Optimizer state and learning-rate scheduling

## Why v0.2 follows v0.1

Once gradients can be produced reliably, optimization becomes a separable subsystem. This version teaches that model parameters, optimizer state and learning-rate state are different objects with different lifetimes.

## Mechanisms

- extensible Optimizer base
- parameter groups
- per-parameter state
- optimizer checkpoint state
- SGD momentum/dampening/Nesterov
- Adam
- AdamW
- Adamax
- RMSprop
- Adagrad
- scheduler base class
- StepLR
- MultiStepLR
- ExponentialLR
- CosineAnnealingLR
- LinearLR
- LambdaLR
- ReduceLROnPlateau

## Architectural lesson

```text
Parameter
   │
   ├──────────► gradient
   │
Optimizer
   ├──────────► parameter groups
   └──────────► optimizer state
                         ▲
                         │
                    Scheduler
                    mutates LR
```

---

# v0.3 — Dataset / Sampler / DataLoader + classification

## Why v0.3 follows v0.2

At this point the framework can train, but it still needs a reusable input pipeline. PyTorch separates dataset storage, index selection, batching and collation because each concern evolves independently.

## Target structure

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

## Current mechanisms

- Dataset
- TensorDataset
- Sampler abstraction
- SequentialSampler
- RandomSampler
- BatchSampler
- sampler-driven DataLoader
- MNIST IDX parsing/downloading
- Flatten
- CrossEntropyLoss
- MNIST MLP training
- Loguru logs
- metrics CSV
- loss/accuracy plots
- best/last model checkpoints

## Remaining v0.3 scope

- Subset
- random_split
- SubsetRandomSampler
- WeightedRandomSampler
- explicit `_SingleProcessDataLoaderIter`
- explicit map-dataset fetcher
- stronger custom-collation tests
- resume-training checkpoint containing model + optimizer + scheduler + epoch
- confusion matrix
- prediction visualization

## Explicit non-goals

- no `num_workers` yet
- no worker processes
- no prefetch queues
- no pinned memory

Those belong after the single-process lifecycle is explicit.

---

# 5. Planned core versions

# v0.4 — TensorImpl-like memory model: Storage / stride / views

## Why this comes next

The current Tensor delegates too much memory-layout behavior to NumPy. Real PyTorch Tensors conceptually combine a storage reference with Tensor-specific metadata such as sizes, strides and storage offset. Multiple views may refer to the same underlying storage while presenting different shapes/layouts.

This version should make that model visible.

## Target architecture

```text
Storage
┌──────────────────────────────┐
│ raw contiguous elements ...  │
└──────────────────────────────┘
        ▲              ▲
        │              │
    Tensor A        Tensor B
    shape           shape
    stride          stride
    offset          offset
```

## Required implementation

### v0.4.1 Storage

- explicit `Storage`
- storage ownership/reference semantics
- dtype
- storage offset
- size metadata
- stride metadata
- contiguous-stride generation

### v0.4.2 Views

- transpose
- permute
- basic slicing views
- shared-storage tests
- `is_contiguous()`

### v0.4.3 `view` / `reshape` / `contiguous`

- `view()` only when layout permits metadata-only reinterpretation
- `reshape()` uses a view when possible and materializes a copy when necessary
- `contiguous()` materializes canonical contiguous layout when needed

### v0.4.4 inplace correctness

- version counter
- inplace version bump
- saved-tensor version checks
- educational inplace ops
- error when backward depends on a value modified after forward

## Definition of Done

The implementation and tests must make these statements demonstrable:

```text
transpose may share storage
transpose may become non-contiguous
view can fail on incompatible layout
reshape may copy
contiguous may allocate new storage
inplace mutation can invalidate backward
```

---

# v0.5 — Operator registry and Dispatcher

## Why Dispatcher comes before a serious backend system

In real PyTorch, one logical operator must work across different execution modes and devices. The dispatcher is the routing layer that selects the relevant implementation according to operator registration and dispatch state.

Without this layer, adding CPU/GPU-like backends later would force large `if device == ...` switches throughout Tensor methods.

## Target architecture

```text
Tensor.add(other)
      │
      ▼
 operator schema: aten-like add.Tensor
      │
      ▼
   Dispatcher
      │
   DispatchKeySet
  ┌───┼──────────────┐
  ▼   ▼              ▼
CPU  Autograd    future CUDA-like
kernel wrapper      kernel
```

## Required implementation

### v0.5.1 Operator schema

- operator name
- namespace
- signature metadata
- registration API
- duplicate registration errors

### v0.5.2 Dispatch keys

Educational keys initially:

- `CPU`
- `Autograd`
- `NoGrad`
- reserved `CUDA`/accelerator-like key

### v0.5.3 Dispatcher

- operator registry
- dispatch table
- dispatch-key priority
- redispatch mechanism
- fallback behavior

### v0.5.4 Tensor integration

Refactor selected Tensor methods so they call registered operators rather than directly owning implementations.

Start with:

- add
- mul
- matmul
- relu
- sum

## Critical learning target

Explain why Autograd can act as a dispatch layer around a backend kernel rather than requiring every backend kernel to implement gradient graph logic itself.

---

# v0.6 — Neural-network operator stack and CNN

## Why CNN waits until after Dispatcher

Conv2d is a good test that the operator system can handle a nontrivial kernel. It should be implemented as a registered operator rather than another special Tensor method.

## Required implementation

- Conv2d reference forward using readable loops
- Conv2d backward for input/weight/bias
- im2col educational optimization
- MaxPool2d
- Dropout
- BatchNorm1d / BatchNorm2d
- buffer semantics for running statistics
- CNN MNIST example

## Learning targets

- parameter sharing
- convolution gradient structure
- im2col → matrix multiplication
- parameters vs buffers
- `train()` / `eval()` behavior

---

# v0.7 — Device and backend abstraction

## Goal

Separate Tensor semantics from the concrete numerical backend.

```text
Tensor
  │
  ▼
Dispatcher
  │
  ├── NumPy CPU backend
  └── optional accelerator backend
```

## Required implementation

- Device object
- device metadata on Tensor/Storage
- `.to(device)`
- backend-owned allocation
- backend kernel registration
- CPU reference backend
- optional CuPy accelerator experiment if environment permits
- device mismatch errors

## Later memory topics

- allocator abstraction
- caching allocator concept
- pinned host memory concept
- stream/event model as documentation or later experiment

Do not attempt a production CUDA C++ runtime.

---

# v0.8 — Advanced Autograd engine

## Goal

Move from “backward computes gradients” to understanding graph lifetime and autograd ownership.

## Required implementation

- leaf vs non-leaf gradient semantics
- `retain_grad()`
- hooks
- graph release after backward
- `retain_graph`
- saved tensors in context
- `mark_dirty`-like safety concept
- custom Function example
- optional `create_graph`
- simple second-order derivative example

## Learning targets

- why a second backward can fail after graph release
- why non-leaf gradients are not always retained
- how hooks attach to graph execution
- how higher-order differentiation changes graph construction

---

# v0.9 — Serialization, training runtime and reproducibility

## Goal

Make training state reproducible and resumable before adding distributed execution.

## Required implementation

- model state_dict
- optimizer state_dict
- scheduler state_dict
- RNG state
- epoch/global-step state
- unified checkpoint object
- save/load utility
- resume MNIST/CNN training deterministically
- seed handling across DataLoader/Sampler

## Optional

- simple module/device summary
- callbacks/events
- gradient clipping

---

# v0.10 — Distributed training concepts

## Goal

Teach the DDP model: model replicas compute local gradients, then gradients are synchronized across workers, typically through collective communication such as AllReduce.

## Staged implementation

### v0.10.1 DistributedSampler

- rank
- world size
- deterministic sharding
- epoch-based shuffle seed

### v0.10.2 ProcessGroup abstraction

Educational API:

- rank
- world_size
- all_reduce
- barrier

### v0.10.3 DDP wrapper

- replicated Module wrapper
- gradient-ready hooks
- gradient synchronization
- parameter consistency tests

### v0.10.4 buckets

- gradient bucket grouping
- overlap concept
- simplified bucketed AllReduce simulation

## Scope rule

A CPU/multiprocessing or simulated implementation is sufficient. The goal is to understand DDP architecture, not reimplement NCCL.

---

# v0.11 — Graph IR and eager-to-graph capture

## Why compiler work starts here

PyTorch eager execution is flexible because Python drives operators dynamically. A compiler needs a stable graph representation before it can optimize anything.

## Required implementation

- Node
- Value
- Graph
- operator nodes
- constants
- graph inputs/outputs
- simple tracer for supported Tensor operations
- graph pretty printer
- graph interpreter

## Graph break concept

Unsupported Python behavior should be represented explicitly rather than silently generating incorrect graphs.

Target mental model:

```text
Python function
      ↓
trace supported region
      ↓
Graph IR
      ↓
unsupported behavior
      ↓
graph break
      ↓
resume eager/tracing
```

---

# v0.12 — Dynamo-like guards and graph caching

## Goal

Explain why graph capture must record assumptions about Python values, shapes, dtypes and other runtime state.

## Required implementation

- simple guard objects
- shape guards
- dtype/device guards
- constant guards
- graph cache
- cache hit/miss
- recompile on guard failure
- dynamic-shape discussion/experiment

## Learning target

Compiled code is correct only while the assumptions made during capture remain valid.

---

# v0.13 — AOTAutograd-like forward/backward graph capture

## Goal

Separate eager autograd execution from ahead-of-time generation of a backward graph.

## Required implementation

- trace forward graph
- derive or capture backward graph for a supported operator subset
- explicit saved intermediates
- joint forward/backward representation
- compare eager gradients vs captured gradients

## Important distinction

The project should teach that eager autograd dynamically executes backward nodes, while an AOT approach can expose backward computation to compiler optimization.

---

# v0.14 — Inductor-like lowering and kernel fusion

## Goal

Build a tiny compiler backend rather than pretending to recreate TorchInductor.

## Pipeline

```text
Graph IR
   ↓
pattern analysis
   ↓
fusion groups
   ↓
lowered loop representation
   ↓
Python/NumPy generated kernel
```

## Required demonstrations

- fuse simple pointwise chains
- remove trivial intermediates
- generated execution function
- output parity tests
- basic benchmark harness

Optional later experiment:

- generate Triton for one pointwise kernel if the environment supports it

---

# v0.15 — Performance/runtime internals

Potential topics after correctness architecture is complete:

- allocator abstraction
- caching allocator simulation
- memory planning
- operator fusion cost model
- asynchronous execution concepts
- streams/events model
- profiling events
- DataLoader multiprocessing/prefetch/pinned-memory experiments

These features should not be implemented earlier merely because real PyTorch has them. They become understandable only after Storage, Dispatcher, Device and runtime boundaries exist.

---

# v1.0 — Coherent educational framework

v1.0 is not “feature parity with PyTorch”.

It means the project can coherently demonstrate the major dependency layers:

```text
Tensor API
   ↓
Storage / strides / views
   ↓
Dispatcher / operators
   ↓
CPU + optional accelerator backend
   ↓
Autograd
   ↓
NN / optim
   ↓
DataLoader
   ↓
checkpoint/runtime
   ↓
distributed concepts
   ↓
graph capture
   ↓
AOT backward
   ↓
compiler lowering
```

A reader completing v0.1 → v1.0 should be able to open real PyTorch source/docs and recognize the purpose of major subsystems even when the production implementation is much more sophisticated.

---

# 6. Version iteration algorithm

Every future version must follow the same development algorithm.

## Step 1 — identify one mental model

Each version should answer one major architectural question.

Good:

- “How can multiple Tensor views share storage?”
- “How does an operator choose a backend kernel?”
- “Why does compiled execution need guards?”

Bad:

- “Add 20 unrelated APIs.”

## Step 2 — define the dependency boundary

Before coding, document:

```text
what this subsystem consumes
what it owns
what it returns
what it must not depend on
```

## Step 3 — build the smallest reference implementation

Prioritize:

```text
correctness
  ↓
readability
  ↓
tests
  ↓
architecture
  ↓
optimization
```

Do not optimize before the mechanism is visible.

## Step 4 — test invariants, not only outputs

Examples:

Storage version:

- two views share storage identity;
- transpose changes stride without copying;
- contiguous may allocate;
- version counter changes after inplace mutation.

Dispatcher version:

- correct dispatch key wins;
- missing kernel fails clearly;
- redispatch skips the current layer;
- Autograd wrapper eventually reaches backend kernel.

Compiler version:

- guard hit reuses cache;
- guard failure recompiles;
- unsupported Python creates graph break;
- compiled output equals eager output.

## Step 5 — integration demonstration

Every major subsystem should have one end-to-end example that shows why it exists.

## Step 6 — version documentation

Update:

- `version.md`
- `Plan.md` on main if roadmap changes
- `Agents.md` if engineering invariants change
- architecture docs

## Step 7 — synchronize main

After validation:

```text
active version branch
       ↓
validated commit
       ↓
main updated immediately
```

`main` must not wait until the entire version is “finished” if the active branch contains already-validated, desired latest state.

---

# 7. Compatibility policy

mini-pytorch is educational, so exact PyTorch API compatibility is not mandatory.

Priorities:

1. conceptual compatibility;
2. understandable implementation;
3. correct mathematical behavior;
4. clear error behavior;
5. familiar API shape where it helps learning.

A feature may intentionally be smaller than PyTorch when full compatibility would obscure the mechanism.

Every intentional difference should be documented.

---

# 8. Testing policy

Each subsystem should include four levels when applicable:

```text
unit
 ↓
property/invariant
 ↓
numerical/reference comparison
 ↓
integration
```

Examples:

- Autograd: finite-difference gradcheck
- Optimizer: convergence + exact state behavior
- Storage/view: storage alias/stride invariants
- Conv2d: compare against a trusted NumPy/reference implementation
- Dispatcher: registration/priority/redispatch tests
- DDP: replica parameter equality after synchronization
- compiler: eager vs compiled parity

No version is “done” with only happy-path tests.

---

# 9. Performance policy

This project should deliberately maintain two implementation levels when useful:

```text
reference implementation
          ↓
optimized educational implementation
```

Examples:

- Conv2d loops → im2col
- eager pointwise ops → fused graph kernel
- one-gradient AllReduce simulation → bucketed synchronization

The readable reference version should remain available whenever practical.

---

# 10. Documentation policy

## On `main`

- `Plan.md`: canonical roadmap and sequencing logic
- `Agents.md`: canonical engineering rules
- `docs/ROADMAP.md`: detailed task-level roadmap where useful
- architecture docs: subsystem explanations

## On every `v0.x` / `v1.x` branch

Mandatory:

- `version.md`
- `Agents.md`

`version.md` must state:

1. version identity;
2. historical/base commit;
3. previous version;
4. core difference from previous version;
5. mechanisms added;
6. mechanisms unchanged;
7. important files;
8. learning goals;
9. non-goals;
10. next-version direction.

---

# 11. Sources used to anchor the architecture

This roadmap is conceptually anchored to current official PyTorch documentation and source structure, especially:

- TensorImpl / Storage / size / stride / offset model
- dispatcher and DispatchKey architecture
- torch.autograd graph concepts
- DistributedDataParallel and DistributedSampler separation
- `torch.compile`, TorchDynamo, AOTAutograd/AOTDispatcher and TorchInductor

Useful upstream references:

- https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h
- https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough
- https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html
- https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.html
- https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html

The mini-pytorch implementation must remain original and educational. References are for architecture understanding, not source-code copying.

---

# 12. Current immediate execution order

The next concrete sequence from the current repository state is:

```text
finish v0.3
  │
  ├── WeightedRandomSampler / Subset / random_split
  ├── explicit DataLoader iterator/fetcher
  ├── resumable MNIST checkpoint
  └── integration tests
  │
  ▼
freeze v0.3 learning snapshot
  │
  ▼
create v0.4 from latest main
  │
  ▼
Storage → strides → views → contiguous → version counter
  │
  ▼
sync every validated v0.4 increment back to main
  │
  ▼
v0.5 Dispatcher
```

This order should only change when a newly discovered dependency makes the current sequence conceptually incorrect. Any such change must update this `Plan.md` and explain the reason.