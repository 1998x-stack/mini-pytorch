# mini-pytorch v1.0

## Version identity

- **Branch:** `v1.0`
- **Previous version:** `v0.15`
- **Theme:** Integrated Architecture
- **Status:** active development
- **Package version target:** `1.0.0`
- **Starting validated baseline:** **247 / 247 tests passed**
- **Base main commit:** `56191655f1188134e62e965dc58fdd3dcd3d062b`

v1.0 asks one final question:

> Can the complete mini-pytorch learning stack be traced, explained, exercised and validated as one coherent architecture without adding another major subsystem?

## Core delta from v0.15

v0.15 completed the last planned mechanism:

```text
LoweredProgram
      ↓
profiling / lifetime analysis
      ↓
MemoryPlan
      ↓
ExecutionPlan
      ↓
PlannedExecutor
```

v1.0 does **not** add another mechanism layer. It consolidates the complete system:

```text
all validated 0.x mechanisms
      ↓
architecture audit
      ↓
public API / naming audit
      ↓
cross-layer integration proofs
      ↓
one end-to-end architecture example
      ↓
mini-pytorch ↔ PyTorch concept map
      ↓
documentation reconciliation
      ↓
release-quality snapshot
```

## Required deliverables

### 1. Architecture audit

Create `docs/ARCHITECTURE_V1.md` that describes actual ownership and dependency boundaries for:

```text
Storage / Device
Tensor
Dispatcher / Backend
Autograd
NN / Optimizer / Data
Training Runtime / Checkpoint
Distributed / DDP
Graph capture / Guards / Cache
AOT differentiation
Lowering / Fusion
Runtime profiling / MemoryPlan / ExecutionPlan
```

For every layer document:

- responsibility / owner;
- inputs and outputs;
- allowed dependencies;
- forbidden dependencies;
- important invariants;
- explicit educational simplifications;
- failure boundary.

### 2. Public API / naming audit

Audit exported learning APIs from:

```text
mini_pytorch
mini_pytorch.nn
mini_pytorch.optim
mini_pytorch.data
mini_pytorch.runtime
mini_pytorch.distributed
mini_pytorch.compiler
```

Only small compatibility-preserving cleanup is allowed. No feature-parity sprint.

### 3. Integrated example

Create `examples/19_integrated_architecture.py` that intentionally demonstrates phases rather than pretending every subsystem belongs in one runtime loop:

```text
training phase
  model/data → eager forward/backward → optimizer

runtime state phase
  checkpoint/RNG/progress round-trip

compiler phase
  capture → graph replay/cache → AOT forward/backward

lowering/runtime phase
  lower → ExecutionPlan → PlannedExecutor

verification phase
  eager/reference/planned outputs and gradients agree
```

### 4. Cross-layer integration tests

Add focused tests proving composition across existing boundaries, especially:

```text
Tensor → Dispatcher → Backend → Autograd
Module → Optimizer → Checkpoint / restore
Capture → Guards / Cache → AOT → Lowering → ExecutionPlan
```

Do not duplicate every historical subsystem test.

### 5. PyTorch conceptual mapping

Create `docs/PYTORCH_MAPPING.md` mapping mini-pytorch learning concepts to real PyTorch subsystem families while explicitly documenting simplifications and non-parity.

### 6. Documentation reconciliation

Review and reconcile canonical docs:

```text
README.md
Agents.md
Plan.md on main after integration
docs/ARCHITECTURE.md
docs/LEARNING_PATH.md
docs/ROADMAP.md
MANIFEST.txt
```

Historical version-specific docs may remain historically accurate.

## Frozen architecture entering v1.0

```text
Python API
   ↓
Tensor / Storage / Device
   ↓
Dispatcher / Backend kernels
   ↓
AutogradEngine
   ↓
Module / Optimizer / Data
   ↓
Training Runtime / Checkpoint / RNG
   ↓
Distributed / DDP simulation
   ↓
Graph capture / Guards / Cache
   ↓
AOT forward + backward Graphs
   ↓
Graph analysis / Lowering / Fusion
   ↓
LoweredProgram
   ↓
profiling / lifetime analysis
   ↓
MemoryPlan / ExecutionPlan
   ↓
PlannedExecutor
```

## Invariants that v1.0 must preserve

- Tensor layout remains Storage + shape + stride + offset + Device.
- Dispatcher remains the operator/backend policy boundary.
- Autograd policy stays outside backend numerical kernels.
- eager backward remains owned by `AutogradEngine`.
- checkpoint/runtime logic stays above Tensor/Dispatcher.
- optimizer remains unaware of DDP communication.
- Graph IR does not bake in backend/Autograd dispatch policy.
- GraphBreak never installs partial cache state.
- AOT backward uses explicit saved Values and grad outputs.
- source Graphs are not mutated by lowering passes.
- ExternalOps continue through Dispatcher.
- runtime buffer reuse never violates output/saved-value lifetime.
- older executors remain correctness references for newer execution layers.

## Non-goals

v1.0 must not add:

- CUDA/Triton code generation;
- a new accelerator backend;
- symbolic dynamic-shape compiler;
- production multiprocessing DataLoader;
- NCCL/network DDP;
- production allocator;
- whole-Python compiler;
- PyTorch API feature parity;
- marketing-style benchmark claims.

## Validation policy

Main command:

```bash
python -m unittest discover -s tests -v
```

Additionally run educational examples that do not require external downloads.

Never claim v1.0 complete until the exact release-candidate HEAD and exact frozen HEAD both pass CI.

## Definition of Done

v1.0 freezes only after:

1. `docs/ARCHITECTURE_V1.md` matches actual source boundaries;
2. public APIs/naming are audited and any cleanup is tested;
3. `examples/19_integrated_architecture.py` demonstrates the major phases coherently;
4. focused cross-layer integration tests are green;
5. `docs/PYTORCH_MAPPING.md` clearly distinguishes analogy from parity;
6. canonical docs and MANIFEST are consistent;
7. package version is `1.0.0`;
8. all historical and v1.0 tests pass on exact branch HEAD;
9. branch is frozen after release-candidate validation;
10. exact frozen HEAD passes CI;
11. validated v1.0 tree is integrated into `main`;
12. canonical `main` docs are restored as the integrated-release view.

## After v1.0

There is no automatic `v1.1` roadmap. Any post-v1.0 expansion requires a separately justified learning goal rather than silently continuing the original 0.x feature sequence.
