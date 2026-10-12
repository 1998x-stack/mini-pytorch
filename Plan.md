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
Device / backend
      ↓
Advanced Autograd/runtime
      ↓
Distributed
      ↓
Compiler stack
```

## 2. Completed milestones

### v0.1 — Eager Tensor + Dynamic Autograd ✅

```text
Tensor → Function → dynamic graph → backward → Parameter → Module → Optimizer
```

Implemented:

- Tensor + `requires_grad`
- reverse-mode autodiff
- gradient accumulation
- differentiable operators
- Module / Parameter / Linear / Sequential
- MSELoss
- SGD / Adam
- numerical gradcheck

### v0.2 — Optimizer State + LR Scheduler ✅

```text
Parameter
   ↓
Optimizer state / parameter groups
   ↓
LR Scheduler
```

Implemented:

- enhanced SGD
- Adam / AdamW / Adamax
- RMSprop / Adagrad
- optimizer state serialization
- parameter groups
- StepLR / MultiStepLR / ExponentialLR
- CosineAnnealingLR / LinearLR / LambdaLR
- ReduceLROnPlateau

### v0.3 — Dataset / Sampler / DataLoader + MNIST ✅

```text
Dataset → Sampler → BatchSampler → DataLoader → collate_fn → Tensor batch
```

Implemented:

- Dataset / TensorDataset
- SequentialSampler / RandomSampler
- BatchSampler
- DataLoader
- MNIST IDX loader
- Flatten / CrossEntropyLoss
- MLP MNIST training
- Loguru / metrics / plots / checkpoints

### v0.4 — Tensor Memory Model ✅

```text
Storage + shape + stride + storage_offset + version counter
```

Implemented:

- explicit Storage
- view aliasing
- transpose / permute / slicing
- `view()` vs `reshape()`
- contiguous semantics
- inplace mutation
- shared version counter
- Autograd saved-version checks

### v0.5 — Operator Registry + Dispatcher ✅

```text
Tensor API
   ↓
Operator Schema
   ↓
Dispatcher
   ↓
DispatchKeySet
   ↓
Autograd / NoGrad / CPU
   ↓
redispatch
```

Implemented:

- OperatorSchema
- operator/kernel registry
- DispatchKey / DispatchKeySet
- dispatch priority
- fallback
- redispatch
- CPU numerical kernels
- Autograd wrappers above backend kernels
- dispatched add / mul / matmul / relu / sum

Validated baseline: 80 tests.

### v0.6 — Neural-Network Operator Stack / CNN ✅

```text
nn.Module
    ↓
functional operator
    ↓
Dispatcher
    ↓
Autograd wrapper
    ↓ redispatch
CPU NN kernel
```

Implemented:

- `nn.Conv2d`
- functional `conv2d`
- Dispatcher-backed `aten::conv2d.default`
- readable nested-loop Conv2d kernel
- im2col + matmul Conv2d path
- input / weight / bias gradients
- stride / padding / dilation
- explicit groups=1 boundary
- `nn.MaxPool2d` + backward routing
- `nn.Dropout`
- `BatchNorm1d` / `BatchNorm2d`
- Parameter vs Buffer semantics for BatchNorm state
- CNN MNIST training example with logging/metrics/plots/checkpoints

Validated baseline: **90 tests**.

---

## 3. Current next milestone — v0.7 Device / Backend Abstraction 🎯

### Why v0.7 comes now

v0.5 introduced operator dispatch, and v0.6 proved the Dispatcher can carry non-trivial neural-network operators. The missing architectural dimension is now **where a Tensor lives and which backend owns its allocation/execution**.

Without a device/backend abstraction, a naive implementation would eventually become:

```python
if tensor.device == "cpu":
    ...
elif tensor.device == "cuda":
    ...
```

repeated throughout Tensor methods and operators.

v0.7 should prevent that design before any real accelerator work begins.

### Target architecture

```text
                       Tensor
                         │
          shape / stride / dtype / device
                         │
                         ▼
                       Storage
                         │
                 backend-owned data
                         │
                         ▼
                    logical operator
                         │
                         ▼
                     Dispatcher
                 ┌───────┴────────┐
                 ▼                ▼
               CPU           accelerator-like
             backend             backend
```

### v0.7.1 Device value object

Implement an explicit `Device` concept:

```text
Device("cpu")
Device("meta") or educational accelerator-like backend
```

Requirements:

- canonical string representation;
- equality/hash semantics;
- validation;
- Tensor exposes `.device`;
- Storage records device/backend identity.

### v0.7.2 Backend-owned allocation

Separate allocation policy from Tensor metadata.

Conceptual target:

```text
Backend.allocate(shape, dtype)
Backend.from_numpy(array)
Backend.to_numpy(storage)
```

The exact API may stay small, but Tensor construction should no longer assume that every Storage allocation is directly a NumPy array forever.

### v0.7.3 Device-aware Dispatcher

Extend DispatchKey inference so backend identity participates in routing.

Target:

```text
CPU Tensor
  ↓
DispatchKeySet{Autograd?, CPU}

other-backend Tensor
  ↓
DispatchKeySet{Autograd?, OTHER_BACKEND}
```

Do not encode backend selection as ad hoc conditions inside every operator.

### v0.7.4 Tensor transfer

Introduce an educational:

```python
x.to("cpu")
x.to(other_device)
```

Semantics to define clearly:

- same-device no-op vs copy behavior;
- new Storage on transfer;
- requires-grad policy;
- view/alias behavior after transfer;
- dtype preservation.

### v0.7.5 Device mismatch errors

Operations on incompatible devices should fail before backend math:

```text
cpu tensor + other-device tensor
→ clear device mismatch error
```

Tests should prove this at Dispatcher/operator boundary.

### v0.7.6 Optional accelerator experiment

If the environment supports a lightweight optional backend such as CuPy, it may be explored behind an optional dependency.

However:

- v0.7 must pass on ordinary CPU CI;
- accelerator hardware must not be required;
- the architectural abstraction is the milestone, not GPU speed.

A `meta` or simulated backend is acceptable if it makes the mechanism clearer and deterministic.

### Definition of Done for v0.7

Tests should demonstrate:

- Device construction/equality/validation;
- Tensor and Storage expose device identity;
- CPU allocation still behaves exactly as before;
- backend allocation boundary is explicit;
- Dispatcher key inference uses backend identity;
- same operator API selects different backend registrations;
- device mismatch errors are clear;
- `.to(device)` follows documented copy/alias semantics;
- Autograd routing still composes above backend routing;
- Conv2d and basic arithmetic remain correct on CPU;
- full regression suite stays green;
- exact final v0.7 branch-head CI passes.

### Explicit non-goals for v0.7

Do not add yet:

- production CUDA allocator;
- CUDA streams/events;
- NCCL;
- automatic mixed precision;
- distributed training;
- graph compiler work;
- performance claims requiring GPU hardware.

---

## 4. Future milestones

### v0.8 — Advanced Autograd Engine

Planned concepts:

- graph lifetime;
- non-leaf `retain_grad()`;
- hooks;
- `retain_graph`;
- custom Function examples;
- optional `create_graph`;
- simple higher-order gradients.

### v0.9 — Serialization / Training Runtime / Reproducibility

Planned concepts:

- unified model + optimizer + scheduler + RNG checkpoint;
- epoch / global-step state;
- deterministic resume;
- seed management;
- optional gradient clipping/callback concepts.

### v0.10 — Distributed / DDP Concepts

Planned concepts:

- DistributedSampler;
- ProcessGroup abstraction;
- simulated/CPU AllReduce;
- DDP wrapper;
- gradient-ready hooks;
- bucket concepts.

### v0.11 — Graph IR and Eager-to-Graph Capture

Planned concepts:

- Node / Value / Graph;
- supported-op tracing;
- graph interpreter;
- explicit graph breaks.

### v0.12 — Dynamo-like Guards and Graph Cache

Planned concepts:

- shape/dtype/device/constant guards;
- graph cache;
- recompile on guard failure;
- dynamic-shape experiments.

### v0.13 — AOTAutograd-like Forward/Backward Capture

Planned concepts:

- forward graph;
- backward graph;
- saved intermediates;
- eager-vs-captured gradient equivalence.

### v0.14 — Inductor-like Lowering and Kernel Fusion

Planned concepts:

- lower-level loop IR;
- pointwise fusion;
- reduction lowering;
- generated Python/NumPy kernels;
- eager vs compiled correctness/performance comparison.

---

## 5. Version-development protocol

For every future version:

```text
read main/Agents.md + Plan.md
        ↓
create version branch from latest main
        ↓
write branch version.md + Agents.md first
        ↓
implement smallest readable reference mechanism
        ↓
add invariant + error + numerical tests
        ↓
add educational example + architecture docs
        ↓
run full regression
        ↓
exact final branch-head CI green
        ↓
freeze branch
        ↓
integrate validated state into main
        ↓
restore canonical main docs
        ↓
advance this Plan.md
```

## 6. Current directive

`v0.6` is complete and frozen. `main` contains the validated CNN operator stack.

The next branch should be **`v0.7`**, created from current `main`, and should focus on **Device / Backend abstraction** while preserving the Dispatcher and Storage architecture already established.