# mini-pytorch — Canonical Development Plan

> `main` roadmap after the v1.0 integrated release.
>
> **Invariant:** `main` is the newest integrated, validated framework state. Historical version branches remain readable learning snapshots.

# 1. Original architecture roadmap — complete

```text
Tensor / Dynamic Autograd
      ↓
Module / Optimizer
      ↓
Data pipeline
      ↓
Tensor memory model
      ↓
Dispatcher / Backend
      ↓
Advanced Autograd Engine
      ↓
Training Runtime
      ↓
Distributed / DDP
      ↓
Graph IR
      ↓
Guards / Graph Cache
      ↓
AOT-style Graph Differentiation
      ↓
Lowering / Fusion
      ↓
Runtime / Performance Mechanics
      ↓
Integrated Architecture
```

| Version | Theme | Key validated mechanism | Tests |
|---|---|---|---:|
| v0.1 | Dynamic Autograd | Tensor → Function/GradFn → backward | historical |
| v0.2 | Optimizer/Scheduler | parameter groups / state / LR scheduling | historical |
| v0.3 | Data pipeline | Dataset / Sampler / DataLoader | historical |
| v0.4 | Tensor memory | Storage / stride / views / version safety | historical |
| v0.5 | Dispatcher | schema / DispatchKey / redispatch | 80 |
| v0.6 | CNN ops | Conv / Pool / BN / Dropout | 90 |
| v0.7 | Device/Backend | CPU + EDU logical placement | 103 |
| v0.8 | Autograd Engine | lifetime / hooks / retain_graph | 123 |
| v0.9 | Training Runtime | deterministic checkpoint resume | 140 |
| v0.10 | DDP | ProcessGroup / reducer / bucket semantics | 159 |
| v0.11 | Graph IR | Dispatcher-level capture/replay | 176 |
| v0.12 | Guards/Cache | specialization / guard failure / recompile | 192 |
| v0.13 | AOT differentiation | explicit backward Graph + saved-value ABI | 206 |
| v0.14 | Lowering/Fusion | analysis / transforms / LoweredProgram | 221 |
| v0.15 | Runtime/Performance | profiling / lifetime / ExecutionPlan | 247 |
| v1.0 | Integrated Architecture | architecture audit + cross-layer proof | **251** |

The original feature-building sequence is complete.

---

# 2. v1.0 integrated architecture

```text
Python API
   ↓
Tensor / Storage / Device
   ↓
Dispatcher / Backend
   ↓
AutogradEngine
   ↓
Module / Optimizer / Data
   ↓
Runtime / Checkpoint / RNG
   ↓
Distributed / DDP

Python Tensor function
   ↓
Capture / Signatures / Guards / Cache
   ↓
AOT forward/backward Graphs
   ↓
Analysis / Lowering / Fusion
   ↓
LoweredProgram
   ↓
Benchmark / Profiler / Lifetime analysis
   ↓
MemoryPlan / ExecutionPlan
   ↓
PlannedExecutor
```

Core distinction:

```text
Tensor semantics
!=
operator dispatch
!=
backward execution
!=
training orchestration
!=
distributed synchronization
!=
compiler transformation
!=
runtime resource planning
```

The purpose of v1.0 is that all of these boundaries can now be explained and exercised without inventing another major subsystem.

---

# 3. v1.0 completion evidence

Frozen version branch:

```text
v1.0
7f6ef6ebb5357d7b15954e780455976fbff3f6b3
```

Release gate:

```bash
python -m unittest discover -s tests -v
python examples/19_integrated_architecture.py
```

Validated result:

```text
251 tests / OK
integrated architecture example / PASS
```

Integrated release artifacts include:

```text
docs/ARCHITECTURE_V1.md
docs/API_AUDIT.md
docs/PYTORCH_MAPPING.md
examples/19_integrated_architecture.py
tests/test_v1_integration.py
```

---

# 4. Post-v1.0 mode

There is **no automatic next version**.

The project now defaults to:

```text
study
architecture explanation
bug fixes
historical corrections
benchmarks
experiments that do not change the release contract
```

Creating a new version requires a new architectural question.

---

# 5. Gate for any future roadmap

Before a new version exists, write a proposal that answers six questions.

## 5.1 Missing mental model

What important framework mechanism cannot be understood from v1.0?

Bad justification:

```text
PyTorch has feature X, so mini-pytorch should also have X.
```

Good justification:

```text
There is an ownership/runtime/compiler concept that v1.0 cannot currently demonstrate.
```

## 5.2 Owning layer

Where should the new mechanism live?

```text
Tensor / Storage
Dispatcher / Backend
Autograd
Data / Runtime
Distributed
Compiler
Planned runtime
```

If ownership is unclear, the feature is not ready for implementation.

## 5.3 Minimal executable proof

Define the smallest implementation that teaches the concept.

Prefer:

```text
readable reference implementation
      ↓
explicit invariants
      ↓
small integration proof
```

before optimization or broad API coverage.

## 5.4 Correctness oracle

Identify the reference behavior.

Examples:

```text
NumPy numerical reference
finite differences
eager Tensor execution
GraphInterpreter
LoweredExecutor
existing runtime/checkpoint behavior
```

A new optimized layer must not become its own unquestioned oracle.

## 5.5 Non-goals

List adjacent features explicitly excluded from the proposed version.

This is required to preserve the one-dominant-mental-model rule.

## 5.6 Validation plan

Specify:

```text
unit tests
invariant/property tests
error paths
reference/numerical proof
cross-layer integration
example/docs
exact-head CI
```

Only after all six sections are credible should `Plan.md` be changed to introduce a new active milestone.

---

# 6. Possible research directions — not roadmap commitments

The following remain deliberately outside v1.0. They are examples of possible future research questions, **not queued versions**:

### Accelerator execution

Potential question:

> What additional ownership boundaries appear when asynchronous accelerator allocation, streams and kernels become real rather than logical-device simulation?

Could involve CUDA/CuPy/Triton experiments, but only if the learning goal is sharper than “add GPU support”.

### Dynamic/symbolic shapes

Potential question:

> How do guards, shape symbols, specialization and memory planning change when concrete shapes are no longer fixed?

### DataLoader concurrency

Potential question:

> How should worker lifecycle, queues, ordering, prefetch and pinned host memory be separated from Dataset/Sampler semantics?

### Real distributed transport

Potential question:

> Which DDP invariants survive once ProcessGroup collectives become multi-process/asynchronous/networked operations?

### Allocator/runtime memory system

Potential question:

> How does a caching allocator change the distinction between logical Value lifetime, reusable compiler buffers and physical device allocation lifetime?

### Higher-order autodiff / functional transforms

Potential question:

> What changes when backward itself must be differentiable and mutation/aliasing must be functionalized?

These topics should stay in research-notes status unless one passes the future-roadmap gate.

---

# 7. Stable maintenance priorities

Without a new roadmap, the preferred work order is:

```text
1. fix correctness bugs
2. strengthen invariant tests
3. improve architecture explanations
4. reconcile stale historical documentation
5. add measurement/benchmark evidence
6. simplify confusing implementation without hiding mechanisms
```

Avoid feature-count growth for its own sake.

---

# 8. Documentation ownership after v1.0

Canonical current-state docs:

```text
README.md                 entry point
Agents.md                 engineering contract
Plan.md                   roadmap / future-change gate
docs/ARCHITECTURE_V1.md   full architecture ownership audit
docs/LEARNING_PATH.md     recommended study sequence
docs/PYTORCH_MAPPING.md   upstream concept-family mapping
docs/ROADMAP.md           completed historical sequence
```

Historical branch docs remain version-specific snapshots.

---

# 9. Release/integration workflow for any justified future version

```text
validated main
   ↓
update Plan.md with new mental model
   ↓
create vX.Y from exact main HEAD
   ↓
write branch version.md + Agents.md first
   ↓
implement reference mechanism
   ↓
tests + docs + example
   ↓
exact release-candidate HEAD CI green
   ↓
freeze branch
   ↓
exact frozen HEAD CI green
   ↓
fast-forward into main
   ↓
restore canonical main docs
   ↓
exact main HEAD CI green
```

This workflow is itself part of the project's architecture discipline.

---

# 10. Current directive

```text
v0.1 → ... → v0.15 → v1.0  ✅ complete
```

The project is now in a stable integrated-learning state.

Do not create `v1.1` until a new missing mental model has been explicitly identified, scoped and justified.
