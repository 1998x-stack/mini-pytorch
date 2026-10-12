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
Advanced Autograd engine
      ↓
Training runtime / reproducibility
      ↓
Distributed
      ↓
Compiler stack
```

The project should preserve earlier mental models while adding one new dominant mechanism per version.

---

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
- fallbacks
- redispatch
- backend numerical kernels
- Autograd wrappers above backend kernels
- dispatched add / mul / matmul / relu / sum

Validated baseline: **80 tests**.

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
- Parameter vs Buffer semantics
- CNN MNIST training example

Validated baseline: **90 tests**.

### v0.7 — Device / Backend Abstraction ✅

Core mental model:

```text
                             Tensor
                shape / stride / dtype / device
                               │
                               ▼
                             Storage
                    backend-owned allocation
                               │
                               ▼
                           Dispatcher
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
           CPU                                  EDU
```

Implemented:

- immutable `Device` value object
- CPU and deterministic simulated `edu` device identities
- Backend abstraction and registry
- `Backend.allocate / from_numpy / to_numpy`
- backend-owned Storage
- `Storage.device / Storage.backend`
- `Tensor.device`
- Tensor construction on a chosen device
- differentiable `Tensor.to(device)`
- same-device no-op / forced-copy semantics
- cross-device Storage materialization and alias break
- device-aware direct `Function.apply`
- `DispatchKey.EDU`
- device-aware Dispatcher key inference
- explicit mixed-device errors
- scalar promotion onto owning Tensor device
- CPU + EDU registrations for core dispatched operators
- CPU + EDU Conv2d / MaxPool2d routing
- recursive `Module.to(device)` for Parameters/Buffers
- functional temporary tensors inherit owning device
- Device/Backend architecture docs and walkthrough example

Important design decision:

```text
EDU backend
≠ CUDA
≠ GPU emulator
```

EDU intentionally uses NumPy host arrays. It validates backend ownership and routing independently of hardware availability.

Validated baseline: **103 tests**.

---

# 3. Current next milestone — v0.8 Advanced Autograd Engine 🎯

## 3.1 Why v0.8 comes after Device / Backend

By v0.7 the framework already has stable boundaries for:

```text
Tensor identity
Storage / aliasing
Operator routing
NN operators
Device / Backend placement
```

This means Autograd can now be deepened without simultaneously redesigning Tensor placement or backend selection.

The current engine is intentionally simple:

```text
forward
   ↓
GradFn graph stays attached to Tensors
   ↓
backward DFS
   ↓
NumPy gradients
```

Important real-framework concepts are still missing:

- graph lifetime;
- released saved intermediates;
- repeated backward rules;
- `retain_graph`;
- retained gradients for non-leaf Tensors;
- gradient hooks;
- removable hook handles;
- richer custom Function examples;
- groundwork for differentiable backward / higher-order gradients.

The central v0.8 question is:

> How does an Autograd engine manage the lifetime and observation of a dynamic backward graph rather than simply traversing a permanent DAG?

## 3.2 Target architecture

```text
                       forward Tensor graph
                               │
                               ▼
                         GradFn / Context
                               │
                               ▼
                         Autograd Engine
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
       gradient flow      graph lifetime      gradient hooks
             │                 │                 │
             ▼                 ▼                 ▼
     leaf accumulation   release/retain     observe/transform
             │
             ▼
      optional non-leaf
        retained grad
```

Placement remains orthogonal:

```text
Autograd Engine
      │
      ├── CPU Tensor graph
      └── EDU Tensor graph
```

Do not reintroduce backend-specific conditions into Autograd.

---

## 3.3 v0.8.1 — Graph lifetime and release semantics

### Problem

Today a Tensor's `GradFn` and its Context remain usable indefinitely. That hides why real PyTorch commonly raises an error when backward is attempted a second time after saved intermediates were freed.

### Implement

Add explicit backward-state lifecycle to Context/GradFn/engine.

Possible educational model:

```text
Context
├── saved_tensors
├── saved_versions
└── released flag
```

After successful backward with default behavior:

```text
saved backward state
        ↓
release()
        ↓
saved_tensors unavailable
```

Requirements:

- saved intermediates can be released;
- version-check metadata can be released consistently;
- attempting to reuse released backward state fails with a clear message;
- leaf Tensors and user Tensor values remain valid;
- graph release must not corrupt unrelated branches.

### Tests

- first backward succeeds;
- second backward through released saved state fails clearly;
- graph with shared nodes releases once safely;
- leaf gradient accumulation semantics remain compatible;
- CPU and EDU graphs behave identically.

---

## 3.4 v0.8.2 — `retain_graph`

Extend:

```python
Tensor.backward(..., retain_graph=False)
```

Semantics:

```text
retain_graph=False
→ release backward state after use

retain_graph=True
→ preserve backward state for reuse
```

Required demonstrations:

```python
loss.backward(retain_graph=True)
loss.backward()
```

should be possible and accumulate leaf gradients.

After the final non-retained backward, another backward should fail when it needs released intermediates.

### Important boundary

`retain_graph` should preserve the backward graph/state, not magically reset accumulated `.grad` values.

Tests should explicitly separate:

```text
graph retention
vs
grad accumulation
```

---

## 3.5 v0.8.3 — Non-leaf `retain_grad()`

Current behavior deliberately discards non-leaf `.grad`.

Add:

```python
intermediate.retain_grad()
```

Mental model:

```text
non-leaf Tensor
     │
     ├── default → transient local grad only
     └── retain_grad() → copy/accumulate into tensor.grad
```

Requirements:

- leaf behavior unchanged;
- non-leaf `.grad` remains `None` by default;
- retained non-leaf grad is populated after backward;
- repeated backward accumulates retained grad consistently;
- detached/non-requires-grad behavior is defined clearly.

Tests:

- default non-leaf grad absent;
- retain_grad captures correct value;
- branch/shared-node gradients accumulate correctly;
- works on EDU as well as CPU.

---

## 3.6 v0.8.4 — Gradient hooks

Introduce a small PyTorch-like Tensor hook concept:

```python
handle = tensor.register_hook(fn)
handle.remove()
```

Hook should receive the gradient flowing through the Tensor.

Educational semantics:

```text
incoming gradient
      ↓
hook_1
      ↓
hook_2
      ↓
possibly transformed gradient
      ↓
accumulate / backward function
```

Requirements:

- deterministic registration order;
- a hook may observe gradient;
- optionally returning a replacement gradient transforms downstream flow;
- `None` means leave gradient unchanged;
- removable handle;
- hook exceptions surface clearly;
- mutation of the original internal gradient buffer should be avoided/documented.

Tests:

- hook sees expected gradient;
- transformed gradient changes upstream leaf grad;
- multiple hooks run in order;
- handle removal works;
- hook works for retained non-leaf Tensor;
- CPU/EDU semantics match.

---

## 3.7 v0.8.5 — Custom Function workflow

`Function` already exists internally, but v0.8 should make the educational custom-operation workflow explicit.

Add a documented example similar to:

```python
class SquarePlusOne(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x + 1

    @staticmethod
    def backward(ctx, grad_output):
        ...
```

Clarify:

```text
forward numerical computation
        ↓
Context saved state
        ↓
backward vector-Jacobian product
```

Required tests/example:

- custom Function forward;
- gradient correctness;
- saved-state lifetime follows retain_graph;
- saved-version safety still catches inplace mutation;
- custom Function works with EDU placement through device-preserving `Function.apply`.

---

## 3.8 v0.8.6 — Higher-order-gradient foundations

True `create_graph=True` requires backward computations themselves to be represented as Tensor operations rather than only NumPy arrays. The current engine does not yet have that architecture.

Therefore v0.8 should **not fake full higher-order autograd**.

Instead, document and optionally prototype the boundary:

```text
current backward
→ np.ndarray gradients
→ graph stops here

future differentiable backward
→ Tensor gradients
→ backward graph can itself be differentiated
```

Potential v0.8 scope:

- add explicit `create_graph` argument that raises a clear `NotImplementedError`, if useful for teaching;
- document what would have to change;
- identify which backward formulas are currently NumPy-only;
- add architecture notes preparing a later higher-order implementation.

Do not claim second derivatives are supported unless numerical tests prove them.

---

## 3.9 Suggested source refactor

If complexity justifies it, introduce an engine module:

```text
mini_pytorch/autograd.py
        │
        ├── Context
        ├── Function
        └── GradFn

mini_pytorch/autograd_engine.py
        │
        ├── graph traversal
        ├── local gradient accumulation
        ├── hooks
        ├── retained grad
        └── graph release
```

Do not split files only for aesthetics. The purpose is to make lifecycle policy visible and keep Tensor from becoming a monolith.

---

## 3.10 v0.8 integration example

Add an educational example such as:

```text
examples/11_autograd_engine.py
```

It should demonstrate in one small program:

1. non-leaf grad absent by default;
2. `retain_grad()`;
3. gradient hook observation/transformation;
4. first backward;
5. repeated backward failure after release;
6. repeated backward success with `retain_graph=True`;
7. same logic on CPU and EDU where practical.

No large training workload is required; v0.8 is about engine semantics.

---

## 3.11 v0.8 documentation

Add:

```text
docs/AUTOGRAD_ENGINE.md
```

It should explain:

- dynamic forward graph vs backward execution;
- leaf vs non-leaf grad storage;
- transient local gradient vs `.grad`;
- graph lifetime;
- `retain_graph`;
- hooks;
- saved tensors;
- version checks;
- why full higher-order gradients need differentiable backward formulas.

Update:

```text
README.md
version.md
Agents.md
```

for the v0.8 branch.

---

## 3.12 Definition of Done for v0.8

Before freeze, tests must demonstrate:

- default backward releases required saved state;
- repeated backward after release fails clearly;
- `retain_graph=True` permits graph reuse;
- graph retention is independent of gradient accumulation;
- non-leaf `.grad` is absent by default;
- `retain_grad()` captures correct gradients;
- hooks observe gradients;
- hook replacement modifies upstream gradient flow;
- hook handle removal works;
- saved-version inplace safety still works;
- custom Function example follows the same lifecycle rules;
- CPU and EDU graphs share the same engine semantics;
- previous 103-test regression remains green;
- architecture docs and example exist;
- exact final v0.8 branch-head GitHub Actions passes.

### Explicit non-goals for v0.8

Do not add yet:

- parallel Autograd scheduling;
- thread pools;
- CUDA streams/events;
- distributed autograd;
- production anomaly detection;
- full `create_graph=True` unless genuinely implemented and tested;
- JVP/forward-mode Autograd;
- DDP;
- compiler capture.

---

# 4. Planned future milestones

## v0.9 — Serialization / Training Runtime / Reproducibility

Planned concepts:

- unified model + optimizer + scheduler checkpoint;
- RNG state;
- epoch/global-step state;
- deterministic resume;
- seed management;
- training-state object;
- optional gradient clipping/callback concepts.

Core question:

> What exact state must be captured to stop a training process and reproduce the same continuation later?

## v0.10 — Distributed / DDP Concepts

Planned concepts:

- DistributedSampler;
- ProcessGroup abstraction;
- simulated/CPU AllReduce;
- DDP wrapper;
- gradient-ready hooks;
- gradient buckets;
- synchronization ordering.

Goal: understand architecture, not reimplement NCCL.

## v0.11 — Graph IR and Eager-to-Graph Capture

Planned concepts:

- Node / Value / Graph;
- supported-op tracing;
- graph interpreter;
- explicit graph breaks;
- eager-vs-graph equivalence.

## v0.12 — Dynamo-like Guards and Graph Cache

Planned concepts:

- shape guards;
- dtype/device guards;
- Python constant guards;
- graph cache;
- recompile on guard failure;
- dynamic-shape experiments.

## v0.13 — AOTAutograd-like Forward/Backward Capture

Planned concepts:

- forward graph;
- backward graph;
- saved intermediates;
- eager-vs-captured gradient equivalence;
- functionalized graph boundary where useful.

## v0.14 — Inductor-like Lowering and Kernel Fusion

Planned concepts:

- lower-level loop IR;
- pointwise fusion;
- reduction lowering;
- generated Python/NumPy kernels;
- eager vs compiled correctness/performance comparison.

## v0.15+ — Runtime / Performance Deepening

Possible later topics:

- caching allocator concepts;
- pinned host memory;
- asynchronous transfer model;
- streams/events;
- operator fusion scheduling;
- more realistic accelerator backend experiment;
- mixed precision and loss scaling;
- profiling/instrumentation.

These should be introduced only when the earlier abstraction requires them.

---

# 5. Version-development protocol

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
advance Plan.md
```

Every frozen branch remains a readable snapshot and should not absorb future subsystem work.

---

# 6. Current directive

`v0.7` is complete and frozen. `main` contains the validated Device / Backend implementation with a **103-test** regression baseline.

The next branch must be **`v0.8`**, created from current `main`, and should focus on the **Advanced Autograd Engine**.

Start with the most fundamental missing behavior:

```text
graph lifetime
    ↓
retain_graph
    ↓
retain_grad
    ↓
hooks
    ↓
custom Function lifecycle
    ↓
higher-order-gradient foundations
```

Do not start DDP, real CUDA runtime, or compiler work before this engine lifecycle is understood and validated.