# mini-pytorch — Architecture-Driven Development Plan

> Canonical roadmap for `main`.
>
> **Invariant:** `main` is always the newest integrated, validated state. Version branches are readable learning snapshots. A completed version is validated, frozen, integrated into `main`, then `main` alone advances the project-level roadmap.

## 1. Project purpose

mini-pytorch removes one layer of PyTorch “magic” at a time with readable Python/NumPy implementations.

The roadmap follows architectural dependency order rather than API popularity:

```text
Tensor / Dynamic Autograd
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
Distributed / DDP concepts
      ↓
Graph IR / eager capture
      ↓
Dynamo-like guards + graph cache
      ↓
AOTAutograd-like forward/backward capture
      ↓
Inductor-like lowering + fusion
      ↓
Runtime / performance mechanics
      ↓
v1.0 integrated architecture
```

One version should introduce one dominant mental model and preserve earlier invariants.

---

# 2. Completed milestones

## v0.1 — Eager Tensor + Dynamic Autograd ✅

```text
Tensor
  ↓
Function.apply
  ↓
GradFn / Context graph
  ↓
backward
```

Key mechanisms:

- Tensor / `requires_grad`;
- reverse-mode autodiff;
- dynamic graph construction;
- leaf gradients;
- broadcasting-aware gradients;
- Module / Parameter / Linear / Sequential;
- SGD / Adam;
- numerical gradcheck.

Primary learning question:

> How can ordinary eager Tensor operations dynamically build a graph that later computes gradients automatically?

---

## v0.2 — Optimizer State + LR Scheduler ✅

```text
Parameter.grad
     ↓
Optimizer parameter groups + state
     ↓
LR Scheduler
```

Key mechanisms:

- SGD momentum / dampening / Nesterov;
- Adam / AdamW / Adamax;
- RMSprop / Adagrad;
- parameter groups;
- optimizer `state_dict` / `load_state_dict`;
- StepLR / MultiStepLR / ExponentialLR;
- CosineAnnealingLR / LinearLR / LambdaLR;
- ReduceLROnPlateau.

Primary learning question:

> Why are optimizer dynamics and learning-rate time state independent from the model weights themselves?

---

## v0.3 — Dataset / Sampler / DataLoader + MNIST ✅

```text
Dataset
  ↓ indices
Sampler
  ↓ batches of indices
BatchSampler
  ↓ samples
DataLoader
  ↓
Tensor batches
```

Key mechanisms:

- Dataset / TensorDataset;
- SequentialSampler / RandomSampler;
- BatchSampler;
- DataLoader;
- MNIST IDX reader;
- Flatten / CrossEntropyLoss;
- MLP MNIST training;
- Loguru training logs;
- metrics CSV;
- loss / accuracy plots;
- example checkpoints.

Primary learning question:

> Why are dataset storage, sample ordering, batching and collation separate abstractions?

---

## v0.4 — Tensor Memory Model ✅

```text
Tensor metadata
├── shape
├── stride
└── storage_offset
       │
       ▼
     Storage
       │
       └── VersionCounter
```

Key mechanisms:

- explicit Storage;
- views / aliasing;
- transpose / permute / slicing;
- negative stride;
- `view()` vs `reshape()`;
- contiguous semantics;
- inplace mutation;
- shared version counter;
- saved-version Autograd safety.

Primary learning question:

> Why is a Tensor more than a NumPy array, and why can two Tensors represent different logical views of the same Storage?

---

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
   ↓
backend kernel
```

Key mechanisms:

- OperatorSchema;
- operator/kernel registry;
- DispatchKey / DispatchKeySet;
- dispatch priority;
- fallback;
- redispatch;
- CPU numerical kernels;
- Autograd wrappers above backend kernels.

Validated baseline: **80 tests**.

Primary learning question:

> Why does a framework route a logical operator through a Dispatcher instead of calling one implementation directly?

---

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

- Conv2d readable nested-loop reference;
- Conv2d im2col path;
- input/weight/bias gradients;
- MaxPool2d;
- Dropout;
- BatchNorm1d / BatchNorm2d;
- Parameter vs Buffer state;
- CNN MNIST example.

Validated baseline: **90 tests**.

Primary learning question:

> Can the Dispatcher architecture support non-trivial neural-network operators without pushing Autograd into the numerical kernel?

---

## v0.7 — Device / Backend Abstraction ✅

```text
Tensor
shape / stride / dtype / device
      ↓
Storage
      ↓
backend-owned allocation
      ↓
Dispatcher
 ┌────┴────┐
 CPU      EDU
```

Key mechanisms:

- immutable Device;
- Backend registry;
- CPU + deterministic educational `edu` backend;
- backend-owned Storage;
- `Tensor.device`;
- `Tensor.to(device)`;
- differentiable transfers;
- device-aware DispatchKey inference;
- mixed-device failures;
- `Module.to(device)`.

Validated baseline: **103 tests**.

Primary learning question:

> How can the same Tensor/operator API execute on different backends without scattering `if device` logic throughout the framework?

---

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
- topological traversal;
- transient local gradients;
- Context release state;
- repeated-backward failure after release;
- `retain_graph=True`;
- non-leaf `retain_grad()`;
- gradient hooks + removable handles;
- public custom Function teaching path;
- explicit higher-order-gradient limitation.

Validated baseline: **123 tests**.

Primary learning question:

> What does an Autograd execution engine do beyond merely traversing a permanent DAG?

---

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

Key mechanisms:

- framework PCG64 RNG;
- `manual_seed / get_rng_state / set_rng_state`;
- RNG integration for Linear/Conv2d/default Dropout/default RandomSampler;
- TrainingState;
- schema-versioned CheckpointState;
- inspectable JSON + NPZ serialization;
- optimizer nested state encoding without pickle;
- staged checkpoint publication;
- model/optimizer/scheduler restore;
- RNG restore-last semantics;
- target CPU/EDU placement preservation;
- strict schema/shape/group/scheduler failures;
- uninterrupted-vs-resumed deterministic equivalence.

Central proof compares:

```text
loss trajectory
model state
Adam moments/counters
StepLR state/current LR
TrainingState
future framework RNG draws
```

Validated baseline: **140 tests**.

Primary learning question:

> What exact state must be saved so “resume training” means the future computation remains the same rather than merely reloading model weights?

---

## v0.10 — Distributed / DDP Concepts ✅

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
synchronized Parameter.grad
  ↓
Optimizer.step
```

Implemented:

- explicit rank / world-size semantics;
- deterministic in-process ProcessGroup world;
- asynchronous educational CollectiveWork handle;
- collective sequence ordering;
- SUM / MEAN all-reduce;
- operation / shape / dtype collective validation;
- DistributedSampler;
- deterministic `set_epoch`;
- padding / drop-last partition semantics;
- DistributedDataParallel Module wrapper;
- Tensor leaf-gradient hooks reused as readiness signals;
- Reducer;
- reverse-registration gradient buckets;
- flatten / all-reduce / unflatten;
- explicit two-phase synchronization barrier;
- initial replica-state broadcast helper;
- stale-gradient and unused-parameter failure semantics;
- EDU placement preservation.

Central mathematical proof:

```text
rank 0 local mean gradient
        +
rank 1 local mean gradient
        ↓ average
DDP synchronized gradient
```

is compared with:

```text
one reference model
concatenated global batch
mean-reduced loss
        ↓
reference gradient
```

After the same SGD step:

```text
replica 0 parameters
== replica 1 parameters
== global-batch reference parameters
```

Validated baseline: **159 tests**.

Primary learning question:

> How do data partitioning, gradient readiness, bucketization and collective reduction compose so replicated models remain equivalent after optimizer updates?

---

# 3. Current next milestone — v0.11 Graph IR / Eager-to-Graph Capture 🎯

## 3.1 Why Graph IR comes now

Through v0.10, mini-pytorch already has a reasonably complete eager execution stack:

```text
Python Tensor API
      ↓
logical operator
      ↓
Dispatcher
      ↓
Autograd/backend policy
      ↓
numerical execution
```

This means there is finally a stable logical-operator boundary worth observing.

A compiler cannot optimize a Python execution history until that execution is represented as explicit data.

The central v0.11 question is:

> How can an eager sequence of Tensor/operator calls be represented as explicit `Value / Node / Graph` data, then replayed through the existing Dispatcher while preserving eager forward and backward semantics?

v0.11 is therefore an **IR and capture milestone**, not a fusion/performance milestone.

---

## 3.2 Target architecture

```text
                 Python function
                      │
                      ▼
               eager Tensor calls
                      │
                      ▼
                  Dispatcher
                  /        \
                 /          \
                ▼            ▼
        normal eager       capture observer
        execution               │
                │               ▼
                │          Graph IR
                │     ┌─────────┼─────────┐
                │     ▼         ▼         ▼
                │   Values     Nodes    Outputs
                │               │
                └───────────────┘
                                │
                                ▼
                       GraphInterpreter
                                │
                                ▼
                           Dispatcher
                                │
                                ▼
                      fresh eager execution
```

Critical principle:

```text
capture records operator semantics
but does not become a second numerical implementation
```

Replay should reuse the existing operator/Dispatcher stack.

---

# 3.3 v0.11.1 — Explicit IR data model

Introduce a small compiler package, for example:

```text
mini_pytorch/compiler/
├── __init__.py
├── ir.py
├── capture.py
└── interpreter.py
```

Core IR concepts:

```python
Value
Node
Graph
```

### Value

A Value represents one data edge in the graph.

Suggested fields:

```text
id
kind
metadata
constant payload when applicable
```

Possible kinds:

```text
input
constant
intermediate
```

Optional metadata useful for inspection:

```text
shape
dtype
device
```

Do not treat the old eager Tensor Python object itself as the persistent graph identity.

### Node

A Node represents one logical operator invocation.

Suggested fields:

```text
id
op_name
args specification
kwargs specification
output Value ids
```

`op_name` should use the existing logical operator/schema name where possible, e.g.:

```text
aten::add.Tensor
aten::matmul.default
aten::relu.default
```

### Graph

A Graph owns:

```text
input Value ids
ordered Nodes
constant Values
graph output Value ids
```

Required invariants:

- deterministic monotonically increasing IDs;
- topological node order;
- every referenced Value is already defined;
- graph outputs reference existing Values;
- no duplicate Value producer;
- stable human-readable formatting.

---

# 3.4 v0.11.2 — Capture context

Introduce an explicit tracing/capture context rather than a permanent global mode.

Possible public API:

```python
captured = trace(fn, *example_inputs)
```

or an internal context manager used by `trace`:

```python
with capture_graph(...) as capture:
    out = fn(*inputs)
```

Core execution model:

```text
real eager input Tensors
      ↓
execute operator normally
      ↓
obtain real eager output Tensor
      ↓
record logical op invocation
      ↓
map eager Tensor identity → IR Value id
```

The eager execution remains the source of truth during tracing.

Capture must be scoped and cleaned up even when the traced function raises.

Do not leak capture state into unrelated subsequent eager calls.

---

# 3.5 v0.11.3 — Dispatcher capture observer boundary

The Dispatcher is the correct high-level observation point because it already sees:

```text
logical operator name
Tensor arguments
kwargs / attributes
backend-independent operator identity
```

Add the smallest possible observation hook around logical operator dispatch.

Conceptually:

```text
Dispatcher.call(op_name, ...)
      ↓
normal dispatch/execution
      ↓
result
      ↓
if capture active:
    capture.record(op_name, args, kwargs, result)
```

Important:

- capturing must not change which DispatchKey/kernel runs;
- capture must not duplicate Autograd wrapper logic;
- capture observer failures should not silently corrupt eager execution;
- replay should call the Dispatcher again, not backend kernels directly.

This becomes the conceptual bridge from eager runtime to compiler IR.

---

# 3.6 v0.11.4 — Inputs and constants

Tracing needs to distinguish runtime inputs from constants.

Example:

```python
def fn(x, w):
    return (x @ w) + 1.0
```

IR should conceptually represent:

```text
%0 = input x
%1 = input w
%2 = constant 1.0
%3 = aten::matmul.default(%0, %1)
%4 = aten::add.Tensor(%3, %2)
return %4
```

Requirements:

- example Tensor arguments become graph input placeholders;
- Tensor/scalar constants become explicit constants;
- constant payloads are copied/owned by the graph rather than hidden live references where practical;
- kwargs remain explicit and deterministic;
- nested simple argument structures are either deliberately supported or deliberately rejected with a clear error.

Do not silently capture arbitrary mutable Python objects.

---

# 3.7 v0.11.5 — Initial supported operator subset

Do not attempt whole-framework capture immediately.

Start from operators already routed cleanly through the Dispatcher:

```text
aten::add.Tensor
aten::mul.Tensor
aten::matmul.default
aten::relu.default
aten::sum.default
```

Then, if the architecture remains clear, include selected NN operators:

```text
aten::conv2d.default
aten::max_pool2d.default
```

Capture support is a whitelist/contract for this milestone.

An unsupported operator must not disappear from the graph.

It should produce a clear graph-break/capture error such as:

```text
GraphBreak:
operator/path is not capture-supported in v0.11
```

---

# 3.8 v0.11.6 — GraphInterpreter

A captured graph becomes useful only if it can be replayed with new inputs.

Implement a small interpreter:

```python
result = GraphInterpreter(graph).run(*new_inputs)
```

Execution model:

```text
bind graph inputs
      ↓
materialize constants
      ↓
for node in graph.nodes:
    resolve Value references
    reconstruct args / kwargs
    Dispatcher.call(node.op_name, ...)
    bind output Value ids
      ↓
resolve graph outputs
```

Important invariant:

> Replay must create fresh eager Tensor/GradFn objects. IR does not store or reuse the original trace-time Autograd graph.

Therefore replayed graph outputs should naturally participate in the existing Autograd Engine.

---

# 3.9 v0.11.7 — Forward and backward equivalence

Central correctness proof should compare:

### Eager

```text
new inputs
   ↓
fn(inputs)
   ↓
loss
   ↓
backward
```

### Captured replay

```text
same new inputs
   ↓
GraphInterpreter(graph)
   ↓
loss
   ↓
backward
```

Assert equivalence of:

```text
forward outputs
input gradients
parameter/input-like Tensor gradients
```

at appropriate numerical tolerance.

This proves that graph capture/replay did not invent a second semantic system.

---

# 3.10 v0.11.8 — Stable textual IR

Add a deterministic readable format such as:

```text
graph(%0: Tensor, %1: Tensor):
  %2 = aten::matmul.default(%0, %1)
  %3 = aten::relu.default(%2)
  %4 = aten::sum.default(%3, axis=None, keepdims=False)
  return %4
```

The exact syntax can remain educational, but output must be stable for the same graph.

Why this matters:

- humans can inspect what tracing captured;
- tests can assert graph structure;
- later v0.12 guards can refer to graph input metadata;
- later lowering stages have an explicit input representation.

Do not generate Python source code yet.

---

# 3.11 v0.11.9 — Graph-break semantics

Explicitly document what v0.11 does **not** capture.

Potential graph breaks include:

- direct `Function.apply` paths that bypass the capture-supported Dispatcher boundary;
- unsupported logical operators;
- arbitrary NumPy computation on `.data`;
- Python side effects;
- arbitrary mutable objects;
- data-dependent Python control flow;
- dynamic creation of unsupported structures.

For this milestone, prefer a clear error over silently falling back and claiming a complete graph.

A future Dynamo-like layer may support segmented graphs/fallback regions, but v0.11 should make the boundary obvious first.

---

# 3.12 v0.11 tests

Add:

```text
tests/test_graph_ir.py
```

Minimum coverage:

### IR model

- deterministic Value/Node IDs;
- topological node ordering;
- input/output identity;
- explicit constant representation;
- stable textual format;
- invalid graph references rejected.

### Capture

- add/mul capture;
- matmul/relu/sum composite graph;
- kwargs/attributes captured;
- graph input placeholders distinct from constants;
- capture context does not leak after completion;
- capture cleanup after exception.

### Replay

- same example inputs reproduce eager output;
- different compatible input values replay correctly;
- CPU replay;
- EDU replay if the captured operator subset permits it;
- outputs are fresh Tensor objects;
- replay builds a fresh Autograd graph.

### Autograd equivalence

- eager vs graph forward equality;
- eager vs graph input-gradient equality;
- repeated replay creates independent graphs.

### Failure paths

- unsupported operator GraphBreak;
- wrong graph input count;
- incompatible constant/value reference;
- malformed Node or duplicate producer;
- unsupported argument structure fails clearly.

Full historical suite must remain green.

---

# 3.13 v0.11 educational example

Add:

```text
examples/14_graph_ir.py
```

It should demonstrate:

```text
def model(x, w):
    return ((x @ w).relu() + 1.0).sum()

example inputs
      ↓
trace
      ↓
print Graph IR
      ↓
run with new inputs
      ↓
compare eager output
      ↓
backward eager vs replay
      ↓
compare gradients
```

The example should make the difference between:

```text
trace-time eager graph
IR graph
replay-time fresh Autograd graph
```

explicit.

---

# 3.14 v0.11 documentation

Add:

```text
docs/GRAPH_IR.md
```

Explain:

- eager execution history vs persistent graph representation;
- Value / Node / Graph;
- why Dispatcher is a useful capture boundary;
- graph inputs vs constants;
- node argument references;
- graph outputs;
- trace-time eager execution;
- interpreter replay;
- replay-time Autograd graph;
- graph breaks;
- why v0.11 is not yet Dynamo/Inductor.

Version branch must also update:

```text
README.md
version.md
Agents.md
MANIFEST.txt
```

---

# 3.15 Definition of Done for v0.11

Before freeze, v0.11 must demonstrate:

- explicit inspectable `Value / Node / Graph` IR;
- deterministic graph IDs/order/formatting;
- scoped eager capture context;
- Dispatcher observation without semantic fork;
- explicit graph inputs/constants/outputs;
- supported-op tracing;
- explicit unsupported-op graph break;
- graph replay through existing Dispatcher;
- eager-vs-replay forward equivalence;
- eager-vs-replay backward equivalence;
- fresh Autograd graph on every replay;
- CPU behavior preserved;
- EDU behavior preserved where supported;
- no regression to Tensor/Autograd/Dispatcher/DDP/checkpoint semantics;
- docs/example complete;
- exact final `v0.11` branch-head GitHub Actions green;
- frozen tree integrated into `main`.

## Explicit non-goals for v0.11

Do not add yet:

- Dynamo-style guards;
- graph cache;
- recompilation;
- symbolic/dynamic shape system;
- arbitrary Python control-flow capture;
- graph segmentation/fallback regions;
- AOTAutograd backward graph capture;
- functionalization;
- kernel lowering;
- fusion;
- generated source code;
- performance claims.

---

# 4. Future milestones

## v0.12 — Dynamo-like Guards and Graph Cache

Core question:

> When is a previously captured graph still valid for a new set of Python/Tensor inputs?

Target concepts:

```text
captured Graph
     +
input specialization assumptions
     ↓
GuardSet
     ↓
new invocation
     ↓
check guards
 ┌───┴────┐
 pass    fail
  │        │
reuse   recapture/recompile
```

Planned mechanisms:

- Tensor shape guard;
- dtype guard;
- device guard;
- selected scalar/constant guard;
- object-identity/structure guard where educationally useful;
- GuardSet readable explanation;
- graph cache;
- cache hit/miss accounting;
- guard failure reason;
- deterministic recapture;
- basic dynamic-shape experiment demonstrating why overly specific guards cause recompilation.

Non-goals:

- full CPython bytecode Dynamo;
- arbitrary Python frame evaluation hooks;
- production symbolic-shape algebra.

---

## v0.13 — AOTAutograd-like Forward / Backward Capture

Core question:

> How can differentiation itself become graph data rather than an opaque eager Engine execution?

Target concepts:

```text
forward graph
      ↓
identify saved intermediates
      ↓
backward contract
      ↓
captured backward graph
```

Planned mechanisms:

- explicit forward Graph;
- backward Graph representation;
- saved-intermediate interface;
- parameter/input gradient outputs;
- eager Autograd vs captured backward equivalence;
- preparation for differentiable/lowered backward execution;
- possibly introduce functionalized operator forms where mutation blocks graph transformation.

Important dependency:

v0.8 intentionally did not fake `create_graph=True`. v0.13 should revisit differentiable backward only when backward operations can genuinely be represented as Tensor/IR operations.

---

## v0.14 — Inductor-like Lowering and Kernel Fusion

Core question:

> Once high-level operator graphs are explicit, how can they be transformed into lower-level loop/kernel programs?

Target pipeline:

```text
Graph IR
   ↓
normalized/lowered ops
   ↓
Loop IR
   ↓
fusion groups
   ↓
generated NumPy/Python kernel
   ↓
execute
```

Planned concepts:

- lower pointwise operators;
- lower simple reductions;
- explicit loop/index IR;
- pointwise fusion;
- simple reduction fusion boundary;
- generated Python/NumPy kernels;
- eager vs compiled correctness;
- basic timing comparison;
- graph visualization of before/after fusion.

No CUDA/Triton code generation is required for the first educational lowering milestone.

---

## v0.15 — Runtime / Performance Mechanics

Potential concepts after the compiler path is understandable:

- execution planning;
- temporary-buffer lifetime;
- simple memory reuse/planning;
- allocator/caching concepts;
- prefetch/scheduling concepts;
- profiler events;
- graph-level execution timing;
- backend capability selection;
- more realistic CPU optimization experiments.

This milestone should be driven by measured bottlenecks rather than API imitation.

---

## v1.0 — Integrated Architecture Learning Release

v1.0 should not mean “feature parity with PyTorch.”

It should mean the educational architecture is coherent end-to-end:

```text
Tensor / Storage
      ↓
Dispatcher / Backend
      ↓
Autograd Engine
      ↓
Module / Optimizer / Data
      ↓
Training Runtime
      ↓
DDP Concepts
      ↓
Graph Capture
      ↓
Guards / Cache
      ↓
AOTAutograd-like
      ↓
Lowering / Fusion
```

Potential release tasks:

- architecture consistency audit;
- remove accidental cross-layer dependencies;
- one end-to-end learning tutorial;
- diagrams linking mini-pytorch concepts to real PyTorch subsystem names;
- stable examples per subsystem;
- full regression and package smoke tests;
- consolidated `docs/LEARNING_PATH.md`;
- release tag.

---

# 5. Permanent engineering rules

## 5.1 Branch rule

```text
main = newest integrated validated state
```

A new version branch must always start from the latest `main`.

A version branch is frozen only after its exact final HEAD passes GitHub Actions.

After validation:

```text
frozen vX.Y
    ↓ fast-forward/integrate
main
    ↓
restore canonical README / Agents / Plan
    ↓
create next version branch later
```

Historical version branches do not receive future architecture features.

## 5.2 Testing rule

Every subsystem should include:

- happy path;
- invariant/property tests;
- failure paths;
- numerical/reference comparison where relevant;
- integration proof;
- full historical regression.

Current integrated baseline:

```text
v0.10
159 tests
OK
```

## 5.3 Architecture rule

Do not collapse layers merely to reduce code size.

Examples:

```text
Autograd policy
≠ backend numerical kernel

Device
≠ Storage

Dataset
≠ Sampler

Optimizer state
≠ model state

Checkpoint serialization
≠ checkpoint logical state

ProcessGroup collective
≠ Reducer

Graph IR
≠ Dispatcher implementation

capture
≠ compilation optimization
```

## 5.4 Documentation rule

Every version must contain:

```text
README.md
version.md
Agents.md
```

Architecture/public changes should also add/update:

```text
docs/*
examples/*
tests/*
MANIFEST.txt
```

## 5.5 Honesty rule

Do not claim mechanisms that the code does not truly implement.

Examples:

- `edu` is not CUDA;
- in-process ProcessGroup is not NCCL/network distributed execution;
- `create_graph=True` is not supported until backward is truly differentiable;
- v0.9 does not guarantee arbitrary multi-worker mid-epoch resume;
- v0.11 tracing will not be called whole-Python-program capture;
- graph replay is not compilation/fusion by itself.

The project is most useful when every claimed mechanism can be found, read and experimentally verified in the source.
