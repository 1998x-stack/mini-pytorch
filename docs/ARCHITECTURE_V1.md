# mini-pytorch v1.0 Architecture

> This document is the architecture audit for the integrated v1.0 release.
>
> It describes the source tree that actually exists. It is not a proposal for a future framework and it is not a claim of PyTorch implementation parity.

## 1. One-page architecture

```text
Python user code
      │
      ├──────────────── training / eager path ────────────────┐
      │                                                       │
      ▼                                                       ▼
Tensor / Module / Data                                  Runtime state
      │                                                 RNG / Checkpoint
      ▼
logical operator
      ▼
Dispatcher
  ├── Autograd / NoGrad policy
  └── CPU / EDU backend key
      ▼
backend kernel
      ▼
Tensor result + eager GradFn
      ▼
AutogradEngine

Python Tensor function
      │
      └──────────────── compiler path ────────────────────────┐
                                                              ▼
                                                     Dispatcher capture
                                                              ▼
                                                         Graph IR
                                                              ▼
                                                  signature / guards / cache
                                                              ▼
                                                   AOT differentiation
                                                  ┌───────────┴───────────┐
                                                  ▼                       ▼
                                            forward Graph            backward Graph
                                                  │                       │
                                                  └───────────┬───────────┘
                                                              ▼
                                                  analysis / transforms
                                                              ▼
                                                     LoweredProgram
                                                              ▼
                                              profiling / lifetime analysis
                                                              ▼
                                                        MemoryPlan
                                                              ▼
                                                       ExecutionPlan
                                                              ▼
                                                      PlannedExecutor
```

The important rule is that these layers do not collapse into one object. Each layer owns a different decision.

---

## 2. Ownership table

| Layer | Owns | Consumes | Produces | Must not own |
|---|---|---|---|---|
| `Device` / `Backend` | logical placement and allocation/copy mechanism | device descriptor, NumPy arrays | backend allocation | autograd, modules, compiler policy |
| `Storage` | flat allocation + shared mutation version | backend allocation | storage referenced by Tensor views | tensor shape semantics beyond flat storage |
| `Tensor` | shape/stride/offset, requires-grad state, user-visible grad | Storage + operator calls | Tensor views/results | backend selection policy, compiler cache |
| `Dispatcher` | operator schema/key selection/redispatch | logical operator + Tensor arguments | selected kernel/fallback result | Module/Optimizer/DataLoader policy |
| backend kernels | numerical operation | Tensor/array payloads | numerical Tensor result | autograd graph traversal, DDP, compiler cache |
| `AutogradEngine` | eager reverse-mode execution + graph lifetime | output Tensor/GradFn graph | accumulated gradients | optimizer updates, graph compilation |
| `Module` | parameter/module/buffer organization | Tensor operations | structured model state | gradient mathematics |
| `Optimizer` | parameter update state and step rule | `Parameter.grad` | mutated parameters | backward graph execution, DDP transport |
| `Dataset` / `Sampler` / `DataLoader` | sample access, index order, batching/collation | dataset/index policy | batches | model/autograd/optimizer state |
| runtime checkpoint/RNG | reproducibility and training progress state | model/optimizer/scheduler/RNG | serializable/restoreable state | numerical kernel policy |
| DDP / Reducer / ProcessGroup | gradient synchronization model | local gradients + rank/world state | synchronized gradients | optimizer algorithm |
| capture / Graph IR | logical eager dataflow | outer Dispatcher-visible operations | `Graph` | backend/autograd dispatch decisions |
| guards / cache | specialization reuse decision | runtime signature | matching cached Graph or miss | numerical execution |
| AOT differentiation | explicit backward-Graph construction | supported forward Graph | forward/backward Graph pair | eager AutogradEngine replacement |
| lowering / fusion | compiler transforms and lower-level instruction grouping | Graph | `LoweredProgram` | runtime buffer ownership |
| profiling / lifetime analysis | measurement and value liveness | LoweredProgram | profiles/lifetimes | correctness oracle |
| MemoryPlan / ExecutionPlan | safe temporary resource assignment | LoweredProgram + lifetimes | buffer slots and release schedule | Tensor semantic ownership |
| PlannedExecutor | execute a validated plan | ExecutionPlan + runtime inputs | output/saved values/stats | graph transformation policy |

---

# 3. Tensor memory layer

## 3.1 Device and Backend

`mini_pytorch.device.Device` is the logical placement descriptor. `mini_pytorch.backend.Backend` owns allocation/copy mechanics for a device type.

Current backends:

```text
cpu → NumPyBackend
edu → EducationalBackend
```

`edu` intentionally uses host NumPy memory while preserving a distinct device/backend identity. It exists to test placement and dispatch architecture without claiming GPU behavior.

### Boundary

```text
Device
   ↓
backend_for(device)
   ↓
Backend.allocate / from_numpy / to_numpy
```

The backend does not decide whether Autograd is active. That is Dispatcher policy.

## 3.2 Storage

`Storage` owns:

```text
flat backend allocation
backend identity
shared VersionCounter
```

Multiple Tensor views can share one Storage. Therefore mutation identity belongs to Storage, not to an individual view.

## 3.3 Tensor layout

A Tensor is modeled as:

```text
Tensor
├── Storage reference
├── shape
├── stride
├── storage_offset
├── requires_grad
├── grad
└── GradFn (when non-leaf differentiable result)
```

This is why these operations have different semantics:

```text
view       → metadata-only when layout is compatible
reshape    → view when possible, copy when needed
contiguous → materialize contiguous allocation when needed
transpose  → usually shared Storage with changed stride
```

### Failure boundary

Examples of explicit failure:

- invalid shape inference;
- incompatible `view` layout;
- mixed-device operation;
- in-place mutation that invalidates saved backward state.

---

# 4. Logical operators and Dispatcher

The Dispatcher is the architectural crossing point between a logical operator and its implementation.

```text
Tensor.__mul__ / functional op
      ↓
logical operator name
      ↓
Dispatcher.keyset_for(...)
      ↓
DispatchKeySet
      ↓
selected key
      ↓
kernel or fallback
```

Current educational keys:

```text
Autograd
NoGrad
CUDA (reserved only)
EDU
CPU
```

The priority model means policy keys can run before backend keys.

## Redispatch

Autograd/fallback behavior may redispatch after removing the already-handled key. This avoids recursively selecting the same policy layer.

## Compiler observation point

Graph capture observes only the outer `dispatch(...)` boundary. Internal redispatch is deliberately not recorded as additional Graph nodes.

That preserves this mental model:

```text
one logical eager operation
      =
one Graph node
```

---

# 5. Eager Autograd

`Tensor` stores persistent user-visible autograd state; `AutogradEngine` owns transient backward execution.

```text
Tensor result
   ↓
GradFn
├── function
├── parents
└── Context
```

Backward execution:

```text
output Tensor
   ↓
build reachable topology
   ↓
validate saved graph state
   ↓
seed output gradient
   ↓
reverse traversal
   ↓
hooks / retain-grad
   ↓
Function.backward
   ↓
accumulate parent gradients
   ↓
release contexts unless retain_graph=True
```

Important separations:

- graph lifetime is not the same as Tensor lifetime;
- leaf `.grad` ownership is not the same as local gradient flow;
- saved-tensor version safety is not the same as graph release;
- `create_graph=True` remains explicitly unsupported rather than pretending higher-order support.

---

# 6. Module, Optimizer and Data

## Module

`Module` organizes Parameters, child Modules and Buffers. It does not implement differentiation.

```text
Module
├── _parameters
├── _modules
├── _buffers
└── training
```

Parameter registration happens through assignment/registration APIs; recursive `state_dict()` is a model-state concern, not an Autograd concern.

## Optimizer

Optimizer consumes already-computed `Parameter.grad` and owns update state such as momentum/moments.

```text
forward
  ↓
loss
  ↓
backward
  ↓
Parameter.grad
  ↓
Optimizer.step
```

The optimizer does not know whether gradients came from one process, simulated DDP, or a future source.

## Data pipeline

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
batch
```

Index-selection policy is separate from sample access. The v1.0 DataLoader remains intentionally single-process.

---

# 7. Training Runtime and Checkpointing

The runtime package owns state needed to reproduce training progress:

```text
model state
optimizer state
scheduler state
framework RNG state
TrainingState
metadata
```

`CheckpointState` is the logical in-memory representation. File serialization uses inspectable metadata plus NumPy tensor payloads.

v1.0 has one package-version source (`mini_pytorch._version.__version__`), and checkpoint metadata uses that same release version by default.

### Restore order

RNG restoration happens after target objects are reconstructed because constructors may consume random values.

This is runtime orchestration, not Tensor-kernel behavior.

---

# 8. Distributed / DDP simulation

The distributed package is an educational, deterministic, single-process simulation of distributed concepts.

```text
DistributedSampler
ProcessGroup
Reducer
GradientBucket
DistributedDataParallel
```

Core separation:

```text
local backward
   ↓
gradient-ready hooks
   ↓
Reducer / buckets
   ↓
ProcessGroup collective simulation
   ↓
synchronized gradients
   ↓
ordinary Optimizer.step
```

The optimizer remains communication-agnostic.

v1.0 does not claim NCCL, networking, multiprocessing, overlap behavior or production collective semantics.

---

# 9. Graph capture, signatures, guards and cache

## Capture

Capture runs the Python function eagerly and observes Dispatcher-visible logical operations.

```text
Tensor inputs
   ↓
Python function executes normally
   ↓
outer Dispatcher calls observed
   ↓
Value / Node / Graph
```

Capture keeps strong references to observed Tensor objects for the session so Python object-id recycling cannot merge unrelated Graph Values.

Unsupported unobserved Tensor-producing eager operations cause a `GraphBreakError` instead of being silently frozen as constants.

## Signatures and guards

Runtime signatures specialize on relevant Tensor properties such as:

```text
shape
dtype
device
requires_grad
grad mode
```

A guard failure is a cache miss/recompile decision, not a mutation of the existing Graph.

## Graph cache

`CompiledFunction` keeps multiple guarded specializations. On a cache hit, a `GraphInterpreter` replays the matching Graph. On a miss, capture creates a new specialization.

This is an educational specialization cache; it is not Python bytecode/frame evaluation.

---

# 10. AOT-style graph differentiation

The eager AutogradEngine remains the correctness reference. AOT differentiation converts a supported forward Graph into an explicit backward Graph.

```text
forward Graph
   ↓
VJP rules
   ↓
DifferentiationSpec
├── forward inputs
├── forward output
└── saved Value ids
   ↓
backward Graph
```

The forward/backward ABI is explicit:

```text
forward inputs
   ↓
forward Graph
   ↓
output + saved Values
           │
           └─────────┐
                     ▼
              backward Graph
              + grad_output
                     ↓
              input gradients
```

The supported operator subset is intentionally small. Unsupported VJP rules fail explicitly.

---

# 11. Lowering and fusion

Lowering never replaces the source Graph in place. It constructs transformed Graphs plus source-to-lowered Value mapping.

```text
Graph
  ↓
normalize
  ↓
analyze
  ↓
DCE
  ↓
safe constant folding
  ↓
use-aware fusion planning
  ↓
LoweredProgram
```

Current instruction families:

```text
ExternalOp
PointwiseFusion
```

Pointwise fusion is deliberately conservative. Shared values, barriers and unsupported patterns stop fusion instead of risking semantic changes.

ExternalOps continue to execute through Dispatcher.

---

# 12. Runtime measurement and planning

## Measurement first

`benchmark()` and `profile_lowered()` provide evidence. They are not correctness oracles.

Timing rules:

```text
explicit warmup
repeated measurements
median / range
same workload
no absolute CI speed thresholds
```

## Value lifetime

Instruction-level `ValueLifetimeAnalysis` answers when a Graph Value is born and last used. Graph outputs and explicitly retained AOT saved Values are extended through the execution boundary.

## MemoryPlan

Node-level buffer intervals allow conservative reuse only when:

```text
eligible pointwise temporary
AND exact shape match
AND exact dtype match
AND exact device match
AND lifetimes do not overlap
AND not graph output
AND not retained/saved
```

## ExecutionPlan

The central compiler/runtime boundary is:

```text
LoweredProgram
= what work executes

ExecutionPlan
= how temporary resources are assigned while executing that work
```

## PlannedExecutor

Planner-owned `out=` buffers are used only for the safe pointwise subset. ExternalOps remain Dispatcher-owned. Each executor call gets a fresh slot pool so returned values from a previous call cannot be overwritten by the next invocation.

---

# 13. Three complete traces

## 13.1 Eager training trace

```text
DataLoader batch
   ↓
Module.forward
   ↓
Tensor logical operators
   ↓
Dispatcher
   ↓
backend kernels
   ↓
Tensor + GradFn graph
   ↓
loss.backward()
   ↓
AutogradEngine
   ↓
Parameter.grad
   ↓
Optimizer.step
   ↓
checkpoint/RNG/progress state
```

## 13.2 Compiled forward trace

```text
Python Tensor function
   ↓
compile_graph
   ↓
InputSignature
   ↓
GraphCache lookup
   ├── guard hit → GraphInterpreter replay
   └── miss → capture → cache specialization
```

## 13.3 AOT + planned runtime trace

```text
Python Tensor function
   ↓
aot_compile
   ↓
forward Graph + backward Graph
   ↓
lower_graph(each)
   ↓
LoweredProgram(each)
   ↓
build_execution_plan(each)
   ↓
PlannedExecutor forward
   ├── output
   └── retained saved Values
           ↓
PlannedExecutor backward + grad_output
           ↓
input gradients
```

Correctness chain:

```text
Eager Tensor execution
      ==
GraphInterpreter
      ==
LoweredExecutor
      ==
PlannedExecutor
```

for the supported subset, with eager Autograd as the gradient reference.

---

# 14. Failure semantics

v1.0 prefers explicit failure to silent fallback that would teach the wrong mechanism.

Examples:

- unknown operator / missing dispatch kernel → `DispatchError`;
- mixed devices → explicit dispatch/device error;
- unsupported capture dataflow → `GraphBreakError`;
- failed guards → inspectable cache miss;
- unsupported AOT VJP → `AOTDifferentiationError`;
- malformed lowering/runtime state → lowering/planning error;
- missing/invalid checkpoint sections → `CheckpointError`;
- invalid view/layout/in-place state → Tensor/autograd error.

---

# 15. Deliberate v1.0 simplifications

The framework is complete as a learning sequence, not as a production deep-learning runtime.

It deliberately omits:

- real CUDA kernels;
- Triton/code generation;
- symbolic/dynamic-shape compiler;
- multiprocessing/prefetch/pinned-memory DataLoader;
- NCCL/network communication;
- production caching allocator;
- multithreaded Autograd engine;
- full PyTorch operator/API coverage;
- whole-Python bytecode capture;
- general higher-order autodiff;
- production profiler and performance guarantees.

These are boundaries to understand, not hidden TODOs required for the v1.0 learning goal.

---

# 16. Source reading order

For a complete architecture pass:

```text
mini_pytorch/device.py
mini_pytorch/backend.py
mini_pytorch/storage.py
mini_pytorch/tensor.py
mini_pytorch/dispatcher.py
mini_pytorch/operator_registry.py
mini_pytorch/autograd.py
mini_pytorch/autograd_engine.py

mini_pytorch/nn/module.py
mini_pytorch/optim/optimizer.py
mini_pytorch/data/dataloader.py
mini_pytorch/runtime/checkpoint.py
mini_pytorch/distributed/reducer.py

mini_pytorch/compiler/ir.py
mini_pytorch/compiler/capture.py
mini_pytorch/compiler/signature.py
mini_pytorch/compiler/guards.py
mini_pytorch/compiler/cache.py
mini_pytorch/compiler/compiled.py
mini_pytorch/compiler/aot.py
mini_pytorch/compiler/analysis.py
mini_pytorch/compiler/lowering.py
mini_pytorch/compiler/runtime_analysis.py
mini_pytorch/compiler/execution_plan.py
```

Then run:

```bash
python examples/19_integrated_architecture.py
python -m unittest discover -s tests -v
```

The integrated example demonstrates the phase boundaries; the full test suite remains the real regression proof.
