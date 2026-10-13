# mini-pytorch — Architecture-Driven Development Plan

> Canonical roadmap for `main`.
>
> **Invariant:** `main` is always the newest integrated, validated framework state. Historical version branches are readable learning snapshots.

# 1. Completed architecture chain

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
```

The planned 0.x mechanism-building sequence is complete.

| Version | Theme | Key validated mechanism | Tests |
|---|---|---|---:|
| v0.1 | Dynamic Autograd | Tensor → Function → backward | historical |
| v0.2 | Optimizer/Scheduler | groups/state/LR scheduling | historical |
| v0.3 | Data pipeline | Dataset/Sampler/DataLoader | historical |
| v0.4 | Tensor memory | Storage/stride/views/version | historical |
| v0.5 | Dispatcher | schema/keys/redispatch | 80 |
| v0.6 | CNN ops | Conv/Pool/BN/Dropout | 90 |
| v0.7 | Device/Backend | CPU + EDU placement | 103 |
| v0.8 | Autograd Engine | lifetime/hooks/retain_graph | 123 |
| v0.9 | Training Runtime | deterministic checkpoint resume | 140 |
| v0.10 | DDP | reducer/buckets/all-reduce equivalence | 159 |
| v0.11 | Graph IR | eager capture/replay | 176 |
| v0.12 | Guards/Cache | specialization/recompile | 192 |
| v0.13 | AOTAutograd-like | explicit backward Graph | 206 |
| v0.14 | Lowering/Fusion | analysis → transforms → LoweredProgram | 221 |
| v0.15 | Runtime/Performance | profiling → lifetime → ExecutionPlan | **247** |

---

# 2. v0.15 completed architecture

```text
LoweredProgram
      ↓
Benchmark / per-instruction profiler
      ↓
ValueLifetimeAnalysis
      ↓
node-level BufferInterval analysis
      ↓
MemoryPlan
  ├── BufferSpec
  ├── BufferSlot
  └── value → slot
      ↓
ExecutionPlan
      ↓
PlannedExecutor
```

Validated properties:

- benchmark warmup/repeats are explicit;
- no absolute latency threshold is used as a unit-test correctness condition;
- profiler produces one ordered event per lowered instruction;
- instruction-level and node-level lifetime views are explicit;
- graph outputs and AOT retained Values escape temporary lifetime;
- exact shape/dtype/device compatibility gates reuse;
- overlapping lifetimes never share a slot;
- eligible pointwise temporaries may share a physical slot;
- only pointwise fusion add/mul/relu currently use preallocated `out=` execution;
- ExternalOps remain Dispatcher-owned;
- slot pools are isolated across executor calls;
- CPU/EDU numerical/device semantics remain correct;
- AOT saved Values survive planned forward execution;
- planned backward gradients match eager/AOT references;
- capture object-id recycling cannot merge distinct Graph Values;
- performance measurement is reported separately from allocation/fusion metrics.

---

# 3. Current milestone — v1.0 Integrated Architecture 🎯

## 3.1 Why v1.0 is consolidation, not another subsystem

The project already contains all major architectural mechanisms intended for this learning sequence.

Adding another large feature would reduce the clarity of the dependency chain. The next goal is therefore:

> Can the entire system now be audited, explained and exercised as one coherent educational framework?

v1.0 should turn this:

```text
many individually correct version layers
```

into this:

```text
one coherent architecture
+
one coherent learning path
+
one coherent release
```

---

## 3.2 v1.0.1 — Architecture audit

Walk the dependency graph from bottom to top:

```text
Storage / Device
      ↓
Tensor
      ↓
Dispatcher / Backend
      ↓
Autograd
      ↓
NN / Optimizer / Data
      ↓
Training Runtime / DDP
      ↓
Capture / Guards / AOT
      ↓
Lowering / ExecutionPlan
```

For each subsystem verify:

- owner/responsibility;
- inputs/outputs;
- what it may depend on;
- what must not depend on it;
- failure semantics;
- current educational simplifications.

Deliverable:

```text
docs/ARCHITECTURE_V1.md
```

with one top-level dependency diagram and subsystem-specific sections.

---

## 3.3 v1.0.2 — Public API / naming audit

Inspect exported learning APIs from:

```text
mini_pytorch
mini_pytorch.nn
mini_pytorch.optim
mini_pytorch.data
mini_pytorch.distributed
mini_pytorch.runtime
mini_pytorch.compiler
```

Questions:

- are names consistent with the mental model?
- are obsolete aliases still exported?
- are version-specific helpers accidentally public?
- do error messages use consistent terminology?
- do docs/examples import the supported path?

Only small cleanup is allowed. Do not erase historical architecture merely to make the surface look more PyTorch-like.

---

## 3.4 v1.0.3 — One end-to-end integrated example

Create an example that crosses the major layers intentionally.

Suggested structure:

```text
1. create model + data
2. eager forward/backward + optimizer
3. checkpoint/RNG state
4. capture a supported inference/training subfunction
5. guards/cache replay
6. AOT forward/backward graph generation
7. lowering/fusion
8. execution planning
9. PlannedExecutor run
10. compare outputs/gradients with eager reference
11. print architecture/runtime explanation
```

The example does not need to force every subsystem into one unrealistic workflow. It should clearly state which phases are training-time, compiler-time and runtime-time.

Deliverable suggestion:

```text
examples/19_integrated_architecture.py
```

---

## 3.5 v1.0.4 — Cross-layer integration tests

Add a small number of tests that prove the layers compose.

Candidate proofs:

### Tensor → Dispatcher → Backend → Autograd

```text
same logical op
correct CPU/EDU execution
correct gradient
```

### Module → Optimizer → Checkpoint → Resume

```text
train
save
fresh objects
restore
continue deterministically
```

### Capture → Guard → Cache → AOT → Lowering → Plan

```text
capture supported function
cache specialization
AOT backward
lower both graphs
plan both graphs
compare eager/planned outputs + gradients
```

### DDP + model/runtime boundary

Keep existing simulated DDP semantics intact; only add integration coverage if a real gap is found.

Do not duplicate every historical test in a giant v1.0 test.

---

## 3.6 v1.0.5 — PyTorch subsystem mapping

Create a conceptual mapping document, not a claim of implementation parity.

Suggested table:

| mini-pytorch concept | Real PyTorch concept family | Important simplification |
|---|---|---|
| `Storage` + Tensor layout | TensorImpl / Storage | Python/NumPy educational model |
| `Dispatcher` | PyTorch dispatcher / DispatchKey | tiny fixed key set |
| `AutogradEngine` | autograd engine | single-process Python engine |
| `Module` / `Parameter` | `torch.nn.Module` | limited hooks/buffers semantics |
| `DataLoader` | PyTorch DataLoader | single-process |
| `ProcessGroup` / DDP | c10d / DDP | deterministic simulation, no NCCL |
| Graph capture/cache | Dynamo-like concepts | no Python bytecode guards |
| AOT differentiation | AOTAutograd-like concepts | limited VJP registry |
| lowering/fusion | Inductor-like concepts | no Triton/codegen |
| ExecutionPlan | runtime/memory planning concepts | conservative NumPy runtime |

Deliverable:

```text
docs/PYTORCH_MAPPING.md
```

Use upstream concepts as learning references, not API compatibility promises.

---

## 3.7 v1.0.6 — Documentation consistency pass

Reconcile:

```text
README.md
Plan.md
Agents.md
docs/ARCHITECTURE.md
docs/LEARNING_PATH.md
docs/ROADMAP.md
all version-specific docs
```

Requirements:

- no stale "next version" statements in canonical main docs;
- historical docs may remain historically accurate;
- current test baseline is consistent;
- examples are numbered and referenced correctly;
- terminology for Graph/LoweredProgram/ExecutionPlan is consistent;
- performance disclaimers remain explicit.

---

## 3.8 v1.0.7 — Release-quality validation

Before v1.0 freeze:

```text
python -m unittest discover -s tests -v
```

Also manually validate the main educational examples that do not depend on external downloads.

Required release checks:

- exact branch-head CI green;
- package version = `1.0.0`;
- MANIFEST complete;
- README shows v1.0 as integrated release;
- no accidental development/temp branch is used as a base;
- frozen historical branches remain unchanged except permitted corrections.

---

# 4. v1.0 explicit non-goals

v1.0 is not the place to add:

- CUDA/Triton kernels;
- symbolic dynamic-shape compiler;
- production multiprocessing DataLoader;
- NCCL/network DDP;
- production caching allocator;
- whole-Python compiler;
- API feature parity with PyTorch;
- benchmark marketing claims.

If future experimentation continues after v1.0, it should start a new separately justified roadmap rather than silently expanding the original learning sequence.

---

# 5. v1.0 Definition of Done

v1.0 freezes only when all are true:

1. top-level architecture documentation matches the real source dependency boundaries;
2. public APIs/naming have been audited and any cleanup is tested;
3. an integrated example traverses the major eager/compiler/runtime layers;
4. cross-layer integration proofs are green;
5. PyTorch concept mapping is explicit and caveated;
6. canonical docs agree on architecture, test baseline and learning order;
7. all historical tests plus v1.0 tests pass;
8. exact release-head CI is green;
9. branch is frozen and integrated back into main;
10. no unsupported production or performance claim is present.

---

# 6. Engineering sequence

```text
v0.14 Lowering / Fusion                    ✅
      ↓
v0.15 Runtime / Performance Mechanics      ✅
      ↓
v1.0  Integrated Architecture              🎯
```

The default next action is not to add another low-level mechanism. It is to make the existing architecture coherent enough that a reader can trace one Tensor expression from Python API to runtime execution and back through gradients without conceptual gaps.
