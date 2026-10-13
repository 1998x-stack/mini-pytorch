# mini-pytorch — Architecture-Driven Development Plan

> Canonical roadmap for `main`.
>
> **Invariant:** `main` is always the newest integrated, validated state. Version branches are readable learning snapshots.

## 1. Project purpose

mini-pytorch removes one layer of PyTorch “magic” at a time with readable Python/NumPy implementations.

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

---

# 2. Completed milestones

## v0.1 — Eager Tensor + Dynamic Autograd ✅
Dynamic reverse-mode autodiff, Module/Parameter, Linear, SGD/Adam, gradcheck.

## v0.2 — Optimizer State + LR Scheduler ✅
Optimizer families, parameter groups/state, AdamW/RMSprop/Adagrad/Adamax, LR schedulers.

## v0.3 — Dataset / Sampler / DataLoader + MNIST ✅
Dataset/Sampler/BatchSampler/DataLoader separation, MNIST, CrossEntropy, MLP training.

## v0.4 — Tensor Memory Model ✅
Storage, shape/stride/offset, views, contiguous, inplace mutation, VersionCounter.

## v0.5 — Operator Registry + Dispatcher ✅
Schema/kernel registry, DispatchKeySet, Autograd/NoGrad/backend routing, redispatch.

Validated baseline: **80 tests**.

## v0.6 — CNN Operator Stack ✅
Conv2d reference/im2col/backward, MaxPool2d, Dropout, BatchNorm, CNN MNIST.

Validated baseline: **90 tests**.

## v0.7 — Device / Backend ✅
Device, backend registry, CPU/EDU, Tensor.to, device-aware dispatch, Module.to.

Validated baseline: **103 tests**.

## v0.8 — Advanced Autograd Engine ✅
Graph lifetime, retain_graph, retain_grad, hooks, custom Function lifecycle.

Validated baseline: **123 tests**.

## v0.9 — Training Runtime / Checkpoint / RNG ✅
TrainingState, framework RNG, JSON+NPZ checkpoint, deterministic resume proof.

Validated baseline: **140 tests**.

## v0.10 — Distributed / DDP Concepts ✅
DistributedSampler, ProcessGroup, all-reduce, DDP Reducer/buckets, global-batch equivalence.

Validated baseline: **159 tests**.

## v0.11 — Graph IR / Eager-to-Graph Capture ✅

```text
Python eager
   ↓
Dispatcher observer
   ↓
Value / Node / Graph
   ↓
GraphInterpreter
   ↓
Dispatcher replay
```

Implemented stable textual IR, constants, GraphBreak, CPU/EDU replay, forward/backward equivalence.

Validated baseline: **176 tests**.

## v0.12 — Guards / Graph Cache / Recompile ✅

```text
runtime call
    ↓
InputSignature
    ↓
GuardSet
    ↓
GraphCache
 ┌──────┴──────┐
 ▼             ▼
hit           miss
 │              │
replay        capture
```

Implemented:

- TensorSignature / InputSignature;
- exact shape/dtype/device/requires-grad specialization;
- grad-mode specialization;
- inspectable GuardResult/GuardSet;
- multi-entry GraphCache;
- CompiledFunction / compile_graph;
- compile/recompile/hit/miss/GraphBreak stats;
- non-mutating explain();
- CPU/EDU specialization;
- multi-input specialization;
- eager-vs-compiled forward/backward equivalence;
- explicit closure Tensor capture-by-value policy.

Validated baseline: **192 tests**.

Primary learning question:

> When is a captured Graph still valid for a new runtime call, and how can multiple valid specializations coexist?

---

# 3. Current next milestone — v0.13 AOTAutograd-like Graph Differentiation 🎯

## 3.1 Why AOTAutograd comes now

The project now has:

```text
logical forward Graph          v0.11
      +
validity/cache semantics       v0.12
```

But differentiation is still executed by the eager AutogradEngine:

```text
GraphInterpreter forward
      ↓
Tensor / GradFn graph
      ↓
AutogradEngine.backward()
      ↓
Python/NumPy VJP formulas
```

The next question is:

> Can reverse-mode differentiation itself become an explicit graph program?

This is the conceptual bridge toward AOTAutograd-like systems.

## 3.2 Target architecture

```text
                 forward Graph
                      │
                      ▼
              differentiate_graph
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   forward execution        backward Graph
          │                       │
          ├── saved values ───────┤
          │                       │
          ▼                       ▼
     user outputs          grad-output inputs
                                  │
                                  ▼
                          GraphInterpreter
                                  │
                                  ▼
                           input gradients
```

The eager AutogradEngine remains the reference oracle.

---

## 3.3 v0.13.1 — Differentiation metadata

Introduce explicit metadata describing differentiation roles:

```text
DifferentiationSpec
├── forward input ids requiring grad
├── forward output ids to differentiate
├── saved value ids required by backward
└── gradient output/input ordering
```

The ordering must be deterministic and inspectable.

Tests:

- scalar output;
- one/multiple grad-requiring inputs;
- non-grad input exclusion;
- deterministic ordering.

---

## 3.4 v0.13.2 — Backward graph IR boundary

Reuse the existing `Graph` representation rather than inventing an unrelated compiler IR.

Backward graph inputs should explicitly represent:

```text
saved forward values
+
grad outputs
```

Backward graph outputs should represent:

```text
gradients for selected forward inputs
```

This makes the forward/backward ABI visible.

---

## 3.5 v0.13.3 — Tensor-producing VJP formulas

Current eager backward formulas largely operate on NumPy arrays.

A graph-producing differentiation system needs formulas expressible as logical Tensor operations.

Begin with a small op set:

```text
aten::add.Tensor
aten::mul.Tensor
aten::matmul.default
aten::relu.default
aten::sum.default
```

Create an educational VJP registry:

```text
operator
   ↓
Graph VJP rule
```

Do not put AOT rules into backend kernels.

---

## 3.6 v0.13.4 — Reverse graph construction

Process forward nodes in reverse topological order.

Conceptually:

```text
forward:
%z = mul(%x, %y)
%o = sum(%z)

backward seed:
d%o

reverse:
d%z = expand_like(d%o, %z)
d%x = mul(d%z, %y)
d%y = mul(d%z, %x)
```

Gradient contributions to shared values must accumulate explicitly in the backward Graph.

---

## 3.7 v0.13.5 — Saved forward values

Backward formulas may need original forward values.

Examples:

```text
mul backward needs x and y
matmul backward needs operands
relu backward needs activation mask/input
```

Determine which forward Values must cross the forward/backward boundary.

Expose them explicitly instead of hiding them in Python `Context`.

Possible result:

```text
AOTProgram
├── forward_graph
├── backward_graph
├── user_output_ids
├── saved_value_ids
└── grad_input_ids
```

---

## 3.8 v0.13.6 — Forward executor with saved outputs

AOT forward execution should return:

```text
user outputs
+
saved forward values
```

without exposing saved values as user-facing outputs.

A small executor wrapper may bind requested internal Value ids while replaying the forward graph.

Do not duplicate backend kernels.

---

## 3.9 v0.13.7 — Backward executor

Possible API:

```python
program = differentiate_graph(graph)
run = AOTInterpreter(program)

outputs, ctx = run.forward(*inputs)
grads = run.backward(ctx, *grad_outputs)
```

The backward path should execute through existing logical operators/Dispatcher where possible.

---

## 3.10 v0.13.8 — Eager reference equivalence

Central correctness experiment:

```text
same function
same inputs

Path A:
eager forward
→ AutogradEngine.backward
→ eager grads

Path B:
forward Graph
→ AOT forward saved values
→ backward Graph
→ AOT grads
```

Require:

```text
outputs_A == outputs_B
grads_A   == grads_B
```

Test CPU and EDU for the supported op subset.

---

## 3.11 v0.13.9 — Shared-subgraph gradient accumulation

Use a graph like:

```python
shared = x * x
out = (shared + shared).sum()
```

Backward Graph must accumulate the two contributions before propagating through `mul`.

This is the graph-level counterpart of the eager AutogradEngine shared-node test.

---

## 3.12 v0.13.10 — Failure semantics

Fail clearly for:

- unsupported operator VJP;
- non-scalar output without supplied grad output;
- malformed saved-value bindings;
- unsupported output structure;
- incompatible grad-output shape/device.

Do not silently fall back inside a partially built backward Graph.

---

## 3.13 Higher-order gradient boundary

Do **not** claim general higher-order gradients merely because backward is represented as a Graph.

Higher-order support requires:

```text
backward Graph execution
      ↓
produces differentiable Tensor graph
      ↓
that graph itself satisfies differentiation rules
```

Treat that as a later explicit milestone unless fully validated.

---

## 3.14 v0.13 tests

Minimum families:

### Differentiation metadata
- input/output role selection;
- deterministic saved-value ordering.

### VJP rules
- add;
- mul;
- matmul;
- relu;
- sum.

### Integration
- forward output equality;
- eager vs AOT input gradients;
- shared-node accumulation;
- CPU / EDU;
- graph cache specialization feeding AOT program as appropriate.

### Failure
- unsupported op;
- bad grad seed;
- missing saved values.

Historical **192 tests** must remain green.

---

## 3.15 v0.13 Definition of Done

v0.13 is complete when:

- forward/backward graph ABI is explicit;
- small VJP registry exists;
- backward Graph is inspectable;
- saved forward values are explicit;
- AOT forward/backward executor exists;
- eager-vs-AOT outputs/grads match;
- shared-node accumulation works;
- CPU/EDU tests pass;
- docs/example/version README updated;
- exact final branch-head CI green;
- frozen v0.13 integrated into main.

---

# 4. Future milestones

## v0.14 — Lowering / Fusion / Inductor-like Concepts

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

Potential transformations:

- pointwise fusion;
- dead-value elimination;
- safe constant folding;
- simple matmul/pointwise lowering;
- tiny generated Python/NumPy kernel representation.

No production Triton claim.

## v0.15 — Runtime / Performance Mechanics

Potential topics:

- execution planning;
- graph memory lifetime;
- preallocation/reuse;
- allocator/pool concepts;
- asynchronous work abstraction;
- profiling;
- eager vs interpreted vs fused timing.

## v1.0 — Integrated Architecture

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

v1.0 means the major architectural boundaries can be followed end-to-end in a small readable codebase; it does not mean feature parity with PyTorch.

---

# 5. Version governance

Every future `vX.Y` must contain:

```text
README.md
version.md
Agents.md
relevant docs
relevant example
tests
```

Freeze only after exact version-branch HEAD passes GitHub Actions, then integrate into main and advance the roadmap.

`main` must never intentionally remain behind a newer validated version branch.
