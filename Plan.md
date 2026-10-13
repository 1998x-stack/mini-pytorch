# mini-pytorch — Architecture-Driven Development Plan

> Canonical roadmap for `main`.
>
> **Invariant:** `main` is always the newest integrated, validated state. Version branches are readable learning snapshots. A completed version is validated, frozen, integrated into `main`, then only `main` advances the project roadmap.

## 1. Project purpose

mini-pytorch removes one layer of PyTorch “magic” at a time with readable Python/NumPy implementations.

The roadmap follows architectural dependency order:

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
Distributed / DDP Concepts
      ↓
Graph IR / Eager Capture
      ↓
Guards / Graph Cache / Recompile
      ↓
AOTAutograd-like graph differentiation
      ↓
Lowering / Fusion
      ↓
Runtime / Performance
      ↓
Integrated architecture
```

One version should introduce one dominant mental model and preserve earlier invariants.

---

# 2. Completed milestones

## v0.1 — Eager Tensor + Dynamic Autograd ✅

```text
Tensor → Function → dynamic graph → backward → Parameter → Module → Optimizer
```

Core mechanisms:

- Tensor / requires_grad;
- dynamic reverse-mode autodiff;
- gradient accumulation;
- Module / Parameter / Linear / Sequential;
- SGD / Adam;
- numerical gradcheck.

## v0.2 — Optimizer State + LR Scheduler ✅

```text
Parameter → Optimizer groups/state → LR Scheduler
```

Core mechanisms:

- SGD variants;
- Adam / AdamW / Adamax;
- RMSprop / Adagrad;
- parameter groups;
- optimizer state serialization;
- major LR schedulers.

## v0.3 — Dataset / Sampler / DataLoader + MNIST ✅

```text
Dataset → Sampler → BatchSampler → DataLoader → Tensor batch
```

Core mechanisms:

- Dataset / TensorDataset;
- SequentialSampler / RandomSampler;
- BatchSampler / DataLoader;
- MNIST IDX reader;
- CrossEntropyLoss;
- MLP MNIST training.

## v0.4 — Tensor Memory Model ✅

```text
Storage + shape + stride + storage_offset + VersionCounter
```

Core mechanisms:

- views / aliasing;
- transpose / permute / slicing;
- `view()` vs `reshape()`;
- contiguous semantics;
- inplace mutation;
- saved-version Autograd safety.

## v0.5 — Operator Registry + Dispatcher ✅

```text
Tensor API → Operator Schema → Dispatcher → DispatchKey → redispatch
```

Core mechanisms:

- schema/kernel registry;
- DispatchKeySet priority;
- Autograd/NoGrad/backend separation;
- redispatch;
- backend numerical kernels.

Validated baseline: **80 tests**.

## v0.6 — Neural-Network Operator Stack / CNN ✅

```text
nn.Module → functional op → Dispatcher → Autograd wrapper → backend kernel
```

Core mechanisms:

- Conv2d reference + im2col;
- Conv2d gradients;
- MaxPool2d;
- Dropout;
- BatchNorm;
- CNN MNIST.

Validated baseline: **90 tests**.

## v0.7 — Device / Backend Abstraction ✅

```text
Tensor metadata → Storage + Device → Backend → Dispatcher
```

Core mechanisms:

- Device;
- backend registry;
- CPU + deterministic EDU backend;
- Tensor.to(device);
- differentiable transfer;
- device-aware dispatch;
- Module.to(device).

Validated baseline: **103 tests**.

## v0.8 — Advanced Autograd Engine ✅

```text
GradFn / Context → AutogradEngine → VJP / lifetime / hooks
```

Core mechanisms:

- explicit Engine;
- graph lifetime / release;
- retain_graph;
- retain_grad;
- gradient hooks;
- custom Function path.

Validated baseline: **123 tests**.

## v0.9 — Training Runtime / Serialization / Reproducibility ✅

```text
model + optimizer + scheduler + RNG + progress
      ↓
CheckpointState
      ↓
validated deterministic resume
```

Core mechanisms:

- framework RNG;
- TrainingState;
- JSON + NPZ checkpoints;
- optimizer/scheduler restore;
- RNG restore-last;
- uninterrupted-vs-resumed exact proof.

Validated baseline: **140 tests**.

## v0.10 — Distributed / DDP Concepts ✅

```text
DistributedSampler
      ↓
local backward
      ↓
Parameter hooks
      ↓
Reducer / buckets
      ↓
ProcessGroup all_reduce
      ↓
synchronized gradients
```

Core mechanisms:

- rank / world-size;
- in-process ProcessGroup;
- SUM/MEAN all-reduce;
- DistributedSampler;
- DDP wrapper;
- Reducer / gradient buckets;
- explicit synchronization barrier;
- global-batch reference equivalence.

Validated baseline: **159 tests**.

## v0.11 — Graph IR / Eager-to-Graph Capture ✅

```text
Python eager function
      ↓
Dispatcher.dispatch
      ├── normal eager execution
      └── capture observer
              ↓
       Value / Node / Graph
              ↓
       GraphInterpreter
              ↓
          Dispatcher replay
```

Implemented:

- explicit `Value`, `Node`, `Graph` IR;
- graph input / constant / intermediate values;
- TensorConstant capture-by-value;
- structured args / kwargs;
- deterministic Value/Node ids;
- stable textual IR;
- Dispatcher-level outer logical-op observation;
- GraphInterpreter replay through Dispatcher;
- CPU/EDU replay;
- trace NoGrad → replay Autograd behavior;
- GraphBreakError for unsupported dynamic dataflow;
- eager-vs-replay forward equality;
- eager-vs-replay input-gradient equality;
- matmul and conv2d replay tests.

Validated baseline: **176 tests**.

Primary learning question:

> How can eager logical-operator execution be represented as explicit data without duplicating numerical execution or Autograd?

---

# 3. Current next milestone — v0.12 Guards / Graph Cache / Recompile 🎯

## 3.1 Why guards come after Graph IR

v0.11 can capture one eager execution into a reusable Graph object.

But this immediately creates a new correctness problem.

Suppose a graph was captured with:

```text
shape=(2, 4)
dtype=float64
device=cpu
requires_grad=True
grad mode enabled
```

A future call may arrive with:

```text
shape=(8, 4)
```

or:

```text
device=edu
```

or:

```text
requires_grad=False
```

The question is no longer:

> Can we replay a Graph?

It becomes:

> Is this particular Graph valid for this particular runtime call?

That is the conceptual role of **guards**.

Once guards exist, the next question follows naturally:

> If several previously captured graphs exist, can one of them be reused instead of capturing again?

That is the role of a **graph cache**.

---

## 3.2 Target architecture

```text
                    Python callable
                         │
                         ▼
                    runtime inputs
                         │
                         ▼
                    InputSignature
                         │
                         ▼
                       Guards
                         │
                         ▼
                    GraphCache
                 ┌───────┴────────┐
                 ▼                ▼
               HIT               MISS
                 │                │
                 ▼                ▼
        GraphInterpreter        capture
                 │                │
                 │                ▼
                 │           new Graph
                 │                │
                 │                ▼
                 └────────── cache entry
                                  │
                                  ▼
                           future cache hits
```

v0.12 is a **specialization/reuse milestone**, not a Python bytecode interception milestone.

---

# 3.3 v0.12.1 — Runtime Tensor signature

Introduce an immutable signature for one Tensor input.

Suggested fields:

```text
shape
dtype
device
requires_grad
```

Possible API:

```python
TensorSignature.from_tensor(x)
```

A call signature for N graph inputs becomes:

```text
InputSignature(
    tensors=(...),
    grad_enabled=...,
)
```

Why grad mode belongs in the call signature:

```text
same Tensor metadata
+
no_grad enabled/disabled
→ different Autograd execution behavior
```

v0.12 should specialize conservatively rather than pretend to support arbitrary dynamic shapes.

Tests:

- deterministic equality/hash;
- different shape != same signature;
- different dtype !=;
- different device !=;
- different requires_grad !=;
- grad-mode difference !=.

---

# 3.4 v0.12.2 — Guard abstraction

A Guard should answer:

```text
Does this runtime call still satisfy one captured assumption?
```

Possible base shape:

```python
class Guard:
    def check(context) -> GuardResult:
        ...
```

`GuardResult` should be inspectable:

```text
passed
reason
expected
actual
```

Suggested guard types:

```text
TensorShapeGuard
TensorDTypeGuard
TensorDeviceGuard
RequiresGradGuard
GradModeGuard
```

Alternative: one structured TensorSignature guard is acceptable if individual failure reasons remain visible.

Important rule:

```text
guard failure
≠ exception in normal compiled wrapper execution
```

A normal guard failure usually means:

```text
cache miss → capture another specialization
```

Errors should be reserved for malformed internal state.

---

# 3.5 v0.12.3 — GuardSet

Group guards associated with one cached Graph:

```text
GuardSet
├── input guards
└── execution-policy guards
```

Possible API:

```python
result = guard_set.check(*runtime_inputs)
```

Result should explain all or first failure.

Expose a deterministic textual form for education/debugging:

```text
input[0].shape == (2, 4)
input[0].dtype == float64
input[0].device == cpu
grad_enabled == True
```

Do not hide guard logic inside opaque lambda closures.

---

# 3.6 v0.12.4 — Graph cache entry

Introduce a cache entry containing:

```text
GraphCacheEntry
├── graph
├── guards/signature
├── compile_id
├── hit_count
└── metadata
```

A callable may own multiple specializations:

```text
fn
├── entry 0: shape=(2,4), CPU, grad=True
├── entry 1: shape=(8,4), CPU, grad=True
└── entry 2: shape=(2,4), EDU, grad=True
```

This is the key mental transition from:

```text
one function → one graph
```

to:

```text
one function → potentially many guarded graph specializations
```

---

# 3.7 v0.12.5 — GraphCache lookup

Create an ordered cache:

```python
GraphCache.lookup(runtime_signature)
```

For the educational implementation, deterministic linear search is preferable to premature hashing complexity.

Lookup semantics:

```text
for entry in entries:
    evaluate guards
    if pass:
        hit

otherwise:
    miss
```

Expose counters:

```text
hits
misses
compiles
entries
```

Tests:

- first call miss;
- identical second call hit;
- signature change miss;
- original signature later hits original entry;
- counter correctness.

---

# 3.8 v0.12.6 — CompiledFunction wrapper

Introduce a user-facing educational wrapper, e.g.:

```python
compiled = compile(fn)
```

or to avoid shadowing Python built-in:

```python
compiled = compile_graph(fn)
```

Call behavior:

```text
compiled(*inputs)
      ↓
make runtime signature
      ↓
cache lookup
      ↓
HIT?
 ├── yes → GraphInterpreter(entry.graph)
 └── no  → capture(fn, *inputs)
           create GuardSet
           store entry
           return eager trace output
```

The miss path should return the eager output already produced during capture, avoiding an unnecessary immediate second execution.

---

# 3.9 v0.12.7 — Recompile semantics

A new specialization should increase an explicit compile/recompile count.

Example:

```text
call 1: shape (2,4) → compile_id 0
call 2: shape (2,4) → hit 0
call 3: shape (8,4) → compile_id 1
call 4: shape (2,4) → hit 0
```

Expose something like:

```python
compiled.stats()
```

returning:

```text
cache_entries
compile_count
hit_count
miss_count
```

This makes recompilation visible rather than magical.

---

# 3.10 v0.12.8 — Explain cache decisions

Add an educational debug surface:

```python
compiled.explain(*inputs)
```

Possible output:

```text
entry 0: FAIL
  input[0].shape expected (2,4), actual (8,4)

entry 1: PASS
```

or an equivalent structured object.

This is important because guard systems are useful only if users can understand why a graph did or did not match.

---

# 3.11 v0.12.9 — Grad mode specialization

Explicitly test the v0.11 principle that Graph IR stores logical operators while runtime policy remains dynamic.

There are two defensible educational policies:

### Policy A — guard grad mode

```text
capture under grad-enabled
→ replay only under same grad-mode specialization
```

### Policy B — do not guard grad mode

because Dispatcher replay can choose Autograd/NoGrad dynamically.

For v0.12, choose one policy deliberately and document why.

Recommended first implementation: **include grad mode in signature** because it makes specialization assumptions visible and parallels the idea that compiler/runtime behavior may depend on execution context.

Do not claim this exactly reproduces every PyTorch Dynamo guard.

---

# 3.12 v0.12.10 — Constants and closure policy

v0.11 captures pre-existing leaf Tensor constants by value.

v0.12 must explicitly decide cache validity when the original closure Tensor later changes.

Simplest safe policy:

```text
captured Tensor constant value is frozen into the Graph
```

Therefore later mutation of the original Python Tensor does not invalidate the entry because the graph is explicitly value-captured.

Document this as a limitation rather than pretending the graph tracks live Module state.

Full Parameter/module mutation guards belong to a later Module-lifting compiler milestone if added.

---

# 3.13 v0.12.11 — GraphBreak interaction

If capture fails with `GraphBreakError`, the compiled wrapper must not insert a corrupt/partial Graph into cache.

Required behavior:

```text
cache miss
   ↓
capture
   ↓
GraphBreakError
   ↓
no cache insertion
```

For the first educational version, propagating GraphBreakError is acceptable.

Optional later policy:

```text
fallback eager execution
```

but do not silently change behavior without explicit docs/tests.

---

# 3.14 v0.12.12 — Core integration proof

Create a deterministic function:

```python
def fn(x, y):
    return ((x * y) + x).relu().sum()
```

Run:

```text
Call A1: shape S1
→ miss / compile graph 0

Call A2: shape S1
→ hit graph 0

Call B1: shape S2
→ miss / compile graph 1

Call A3: shape S1
→ hit graph 0
```

Verify:

```text
all outputs == eager reference
all input grads == eager reference
compile_count == 2
cache entries == 2
hits/misses correct
```

Also repeat for CPU/EDU specialization as appropriate.

---

# 3.15 v0.12 tests

Minimum test families:

### Signature / Guard

- shape change;
- dtype change;
- device change;
- requires-grad change;
- grad-mode change;
- deterministic representation;
- failure explanations.

### Cache

- cold miss;
- warm hit;
- multiple specializations;
- return to previous specialization;
- stats;
- no insertion after failed capture.

### Semantics

- eager vs cached replay output;
- eager vs cached replay gradients;
- CPU / EDU;
- NoGrad policy;
- Tensor constants;
- historical 176 tests stay green.

---

# 3.16 v0.12 Definition of Done

v0.12 is complete when:

- runtime signature model is explicit;
- Guard/GuardSet is inspectable;
- GraphCache stores multiple specializations;
- compiled wrapper distinguishes hit/miss;
- recapture happens on guard miss;
- stats and explainability exist;
- GraphBreak cannot poison cache;
- eager/cached forward and backward equivalence pass;
- README/version.md/Agents.md/docs/example updated;
- exact final branch-head CI is green;
- frozen v0.12 is integrated into `main`.

---

# 4. Future milestones

## v0.13 — AOTAutograd-like Graph Differentiation

Central question:

> Can backward itself become explicit graph data instead of opaque Python/NumPy VJP execution?

Target concepts:

```text
captured forward Graph
      ↓
functionalized differentiable representation
      ↓
forward graph + backward graph
```

Likely work:

- Tensor-producing backward formulas;
- explicit gradient graph Values;
- saved forward intermediates as graph interface;
- forward/backward graph partition;
- reference equivalence with eager AutogradEngine.

Do not begin before graph specialization/cache semantics are stable.

## v0.14 — Lowering / Fusion / Inductor-like Concepts

Central question:

> Once we have explicit graph programs, how can high-level operators be transformed into fewer lower-level computations?

Target concepts:

```text
Graph IR
  ↓
normalization
  ↓
pattern/fusion analysis
  ↓
lowered loop/tensor IR
  ↓
reference executor
```

Possible educational transformations:

- pointwise op fusion;
- dead-value elimination;
- constant folding where semantically safe;
- simple matmul/pointwise lowering;
- tiny generated Python/NumPy kernel representation.

No production Triton claim.

## v0.15 — Runtime / Performance Mechanics

Potential topics:

- allocator/pool concepts;
- execution planning;
- graph memory lifetime;
- preallocation/reuse;
- asynchronous work abstraction;
- profiling;
- dispatch/capture/cache timing;
- performance comparison of eager vs interpreted vs fused paths.

## v1.0 — Integrated Architecture

Final educational architecture should connect:

```text
Tensor / Storage
      ↓
Dispatcher / Backend
      ↓
Autograd
      ↓
Module / Optimizer / DataLoader
      ↓
Runtime / Checkpoint
      ↓
Distributed
      ↓
Graph Capture
      ↓
Guards / Cache
      ↓
AOTAutograd-like
      ↓
Lowering / Fusion
```

v1.0 is not “feature parity with PyTorch”.

It means:

> the major architectural boundaries can be followed end-to-end in a small readable codebase, and each version branch preserves the point at which one more layer stopped being magic.

---

# 5. Version governance checklist

Every future `vX.Y` must contain:

```text
README.md
version.md
Agents.md
relevant docs
relevant example
tests
```

Freeze only after:

```text
exact version-branch HEAD
      ↓
GitHub Actions
      ↓
full regression green
```

Then:

```text
freeze branch
      ↓
integrate into main
      ↓
restore canonical main docs
      ↓
advance Plan.md
```

`main` must never intentionally remain behind a newer validated version branch.
