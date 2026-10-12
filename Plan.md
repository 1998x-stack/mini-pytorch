# mini-pytorch — Architecture-Driven Development Plan

> Canonical roadmap for `main`.
>
> **Invariant:** `main` is always the newest integrated, validated state. Version branches are readable learning snapshots. A completed version is validated, frozen, then integrated into `main` before the next branch is created.

## 1. Project purpose

mini-pytorch removes one layer of PyTorch “magic” at a time using readable Python/NumPy implementations.

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

Core mental model:

```text
Tensor → Function → dynamic graph → backward → Parameter → Module → Optimizer
```

Implemented:

- Tensor + `requires_grad`
- reverse-mode autodiff
- gradient accumulation
- common differentiable operations
- Module / Parameter / Linear / Sequential
- MSELoss
- SGD / Adam
- numerical gradcheck

### v0.2 — Optimizer State + LR Scheduler ✅

Core mental model:

```text
Parameter
   ↓
Optimizer state / parameter groups
   ↓
LR Scheduler
```

Implemented:

- richer SGD
- Adam / AdamW / Adamax
- RMSprop / Adagrad
- optimizer state serialization
- parameter groups
- StepLR / MultiStepLR / ExponentialLR
- CosineAnnealingLR / LinearLR / LambdaLR
- ReduceLROnPlateau

### v0.3 — Dataset / Sampler / DataLoader + MNIST ✅

Core mental model:

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
- end-to-end MNIST MLP training
- logging / metrics / plots / model checkpoints

### v0.4 — Tensor Memory Model ✅

Core mental model:

```text
Storage + shape + stride + storage_offset + version counter
```

Implemented:

- explicit Storage
- view aliasing
- transpose / permute / slicing
- `view()` vs `reshape()`
- `is_contiguous()` / `contiguous()`
- inplace mutation
- shared Storage version counter
- Autograd saved-version checks

### v0.5 — Operator Registry + Dispatcher ✅

Core mental model:

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

- `OperatorSchema`
- operator/kernel registry
- `DispatchKey`
- `DispatchKeySet`
- dispatch priority
- per-key fallbacks
- redispatch
- CPU numerical kernels
- Autograd wrappers above backend kernels
- NoGrad fallback
- dispatched `add / mul / matmul / relu / sum`
- dispatcher introspection and error paths

Validated with 80-test full regression suite.

---

## 3. Current next milestone — v0.6 Neural-Network Operator Stack / CNN 🎯

### Why v0.6 comes after Dispatcher

The Dispatcher is only meaningful if it can support a nontrivial operator family. `Conv2d` is the first good stress test because it combines:

- multi-dimensional shape rules;
- trainable weights and bias;
- nontrivial backward formulas;
- meaningful backend kernel choices;
- higher computational cost than simple pointwise operations.

Therefore v0.6 should **use** the v0.5 operator architecture rather than bypassing it.

### Target architecture

```text
nn.Conv2d
    ↓
functional conv2d
    ↓
aten::conv2d.default
    ↓
Dispatcher
    ├── Autograd wrapper
    │       ↓ redispatch
    └── CPU reference kernel
            ↓
      readable convolution
```

### Required implementation

#### v0.6.1 Conv2d reference path

- `nn.Conv2d`
- functional `conv2d`
- registered operator schema
- CPU forward reference kernel using clear loops
- input/weight/bias backward
- stride
- padding
- deterministic shape validation
- numerical gradcheck against finite differences / trusted NumPy reference

#### v0.6.2 Conv2d educational optimization

After the loop implementation is correct:

- im2col
- matrix-multiplication path
- reference-vs-im2col equivalence tests
- basic benchmark/example

Keep the readable loop path; do not replace it entirely with optimized code.

#### v0.6.3 Pooling

- MaxPool2d
- Dispatcher registration
- backward index routing
- shape/error tests

#### v0.6.4 Dropout

- `Dropout`
- train/eval behavior
- scaling convention
- deterministic seeded tests

#### v0.6.5 BatchNorm

- BatchNorm1d / BatchNorm2d
- train/eval behavior
- learnable affine parameters
- running mean/variance as buffers
- buffer state_dict tests

#### v0.6.6 CNN MNIST

End-to-end model such as:

```text
Conv2d
  ↓
ReLU
  ↓
MaxPool2d
  ↓
Conv2d
  ↓
ReLU
  ↓
Flatten
  ↓
Linear
  ↓
CrossEntropyLoss
```

The example should reuse existing Dataset/DataLoader/Optimizer/Scheduler infrastructure.

### Definition of Done for v0.6

Tests must demonstrate:

- Conv2d forward correctness;
- Conv2d input/weight/bias gradients;
- Dispatcher routes Conv2d through Autograd → CPU;
- reference loops and im2col agree;
- pooling forward/backward correctness;
- Dropout train/eval semantics;
- BatchNorm parameter vs buffer semantics;
- CNN can execute a training step end-to-end;
- full regression suite remains green;
- exact final `v0.6` branch-head CI passes.

### Explicit non-goals

v0.6 should not add:

- real CUDA;
- `.to(device)`;
- distributed training;
- multiprocessing DataLoader;
- compiler graph capture.

---

## 4. Planned future milestones

### v0.7 — Device and Backend Abstraction

Goal:

```text
Tensor
  ↓
Dispatcher
  ├── NumPy CPU backend
  └── optional accelerator backend
```

Planned concepts:

- `Device`
- device metadata on Tensor/Storage
- `.to(device)`
- backend-owned allocation
- CPU registration
- optional CuPy experiment if environment permits
- device mismatch errors

### v0.8 — Advanced Autograd Engine

Planned concepts:

- graph lifetime
- non-leaf `retain_grad()`
- hooks
- `retain_graph`
- custom Function examples
- optional `create_graph`
- simple higher-order gradients

### v0.9 — Serialization / Training Runtime / Reproducibility

Planned concepts:

- unified model + optimizer + scheduler + RNG checkpoint
- epoch / global-step state
- deterministic resume
- seed management
- optional gradient clipping/callback concepts

### v0.10 — Distributed / DDP Concepts

Planned concepts:

- DistributedSampler
- ProcessGroup abstraction
- simulated/CPU AllReduce
- DDP wrapper
- gradient-ready hooks
- bucket concepts

The goal is architectural understanding, not reimplementing NCCL.

### v0.11 — Graph IR and Eager-to-Graph Capture

Planned concepts:

- Node / Value / Graph
- supported-op tracing
- graph interpreter
- explicit graph breaks

### v0.12 — Dynamo-like Guards and Graph Cache

Planned concepts:

- shape/dtype/device/constant guards
- graph cache
- recompile on guard failure
- dynamic-shape experiments

### v0.13 — AOTAutograd-like Forward/Backward Capture

Planned concepts:

- forward graph
- backward graph
- saved intermediates
- eager-vs-captured gradient equivalence

### v0.14 — Inductor-like Lowering and Kernel Fusion

Planned concepts:

- simple lower-level loop IR
- pointwise fusion
- reduction lowering
- generated Python/NumPy kernels
- eager vs compiled correctness/performance comparison

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
integrate validated state to main
        ↓
advance this Plan.md
```

## 6. Current directive

`v0.5` is complete and frozen. `main` contains the validated Dispatcher implementation.

The next version branch should be **`v0.6`**, created from current `main`, and should focus on the **neural-network operator stack / CNN**, starting with a readable Dispatcher-backed Conv2d reference implementation.