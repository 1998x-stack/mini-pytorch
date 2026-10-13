# mini-pytorch v1.0

## Version identity

- **Branch:** `v1.0`
- **Previous version:** `v0.15`
- **Theme:** Integrated Architecture
- **Status:** frozen integrated learning snapshot
- **Package version:** `1.0.0`
- **Starting baseline:** 247 tests / OK
- **Frozen validation target:** **251 / 251 tests passed + integrated example passed**
- **Base main commit:** `56191655f1188134e62e965dc58fdd3dcd3d062b`
- **Validated release-candidate HEAD:** `7a8391f0959e0671e2bf40720e67296793048947`

v1.0 completes the original mini-pytorch learning sequence.

The final question was:

> Can the complete framework be traced, explained, exercised and validated as one coherent architecture without adding another major subsystem?

The answer is represented by the combination of source, tests, architecture audit, mapping document and integrated example in this frozen branch.

## Core delta from v0.15

v1.0 is a consolidation release:

```text
validated 0.x mechanism chain
      ↓
architecture ownership audit
      ↓
API / naming audit
      ↓
cross-layer executable proofs
      ↓
integrated architecture example
      ↓
PyTorch concept mapping
      ↓
documentation reconciliation
      ↓
release-quality snapshot
```

No new major runtime/compiler subsystem was added.

## Frozen integrated architecture

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

Python Tensor function
   ↓
Graph capture / Signatures / Guards / Cache
   ↓
AOT forward + backward Graphs
   ↓
Analysis / DCE / constant folding / Fusion
   ↓
LoweredProgram
   ↓
Benchmark / profiling / lifetime analysis
   ↓
MemoryPlan / ExecutionPlan
   ↓
PlannedExecutor
```

## Added in v1.0

### Architecture audit

```text
docs/ARCHITECTURE_V1.md
```

Documents actual ownership, dependencies, forbidden dependencies, invariants, failure boundaries and educational simplifications.

### API audit

```text
docs/API_AUDIT.md
```

Conclusion: preserve the broad educational surfaces; do not perform a cosmetic breaking rewrite merely to mimic PyTorch API shape.

### PyTorch mapping

```text
docs/PYTORCH_MAPPING.md
```

Maps mini-pytorch concepts to current PyTorch subsystem families while explicitly rejecting claims of source, API, edge-case or performance parity.

### Integrated executable example

```text
examples/19_integrated_architecture.py
```

Proves these phases compose:

```text
eager training
checkpoint/runtime state restore
guarded graph specialization/cache
eager gradient reference
AOT differentiation
lowering
ExecutionPlan
PlannedExecutor forward/backward
```

### Cross-layer integration tests

```text
tests/test_v1_integration.py
```

Covers:

```text
Tensor → Dispatcher → Backend → Autograd
Module → Optimizer → Scheduler → Checkpoint restore
Capture → Cache → AOT → Lowering → ExecutionPlan
Planned backward → eager gradient reference
```

### Release metadata consistency

Added:

```text
mini_pytorch/_version.py
```

The runtime package and checkpoint layer now agree on release `1.0.0`.

### Canonical documentation reconciliation

Updated:

```text
README.md
docs/ARCHITECTURE.md
docs/LEARNING_PATH.md
docs/ROADMAP.md
MANIFEST.txt
```

The former stale roadmap/learning docs now describe the completed architecture sequence.

### CI governance

GitHub Actions now validates:

```text
main
v*
```

and release CI runs both:

```bash
python -m unittest discover -s tests -v
python examples/19_integrated_architecture.py
```

## Correctness chain

For the supported compiler/runtime subset:

```text
Eager Tensor execution
      ==
GraphInterpreter
      ==
LoweredExecutor
      ==
PlannedExecutor
```

For the supported explicit backward path:

```text
eager AutogradEngine gradients
      ==
AOT backward Graph gradients
      ==
planned backward gradients
```

## Release-candidate validation evidence

Exact release-candidate HEAD:

```text
7a8391f0959e0671e2bf40720e67296793048947
```

passed GitHub Actions with:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
mini-pytorch-learning 1.0.0
```

Unit suite:

```text
Ran 251 tests
OK
```

Integrated example:

```text
checkpoint framework_version = 1.0.0
GraphCache compile_count = 1
GraphCache hit_count = 1
planned output matches eager: PASS
planned gradients match eager: PASS
```

The frozen branch HEAD must pass this same exact-head workflow before integration into `main`.

## Key v1.0 artifacts

```text
mini_pytorch/_version.py

tests/test_v1_integration.py
examples/19_integrated_architecture.py

docs/ARCHITECTURE_V1.md
docs/API_AUDIT.md
docs/PYTORCH_MAPPING.md
```

## Preserved architecture invariants

- Storage owns allocation identity and shared mutation version.
- Tensor owns shape/stride/offset and autograd-facing state.
- Dispatcher owns logical operator policy/backend selection.
- backend kernels do not own Autograd/Module/DDP/compiler policy.
- AutogradEngine owns eager backward execution.
- Module owns parameter/module/buffer organization, not differentiation.
- Optimizer consumes gradients but does not compute or communicate them.
- checkpoint/RNG/progress state remains a runtime concern.
- DDP synchronization remains outside optimizer algorithms.
- Graph IR contains logical operations rather than baked-in backend policy.
- guard failure creates/reuses specialization rather than mutating incompatible Graphs.
- AOT backward uses an explicit saved-value ABI.
- lowering does not mutate source Graphs.
- ExternalOps remain Dispatcher-owned.
- ExecutionPlan owns resource assignment, not Tensor semantics.
- saved/output lifetimes constrain memory reuse.

## Explicit non-goals

The frozen v1.0 release does not claim:

- CUDA/Triton code generation;
- real accelerator execution;
- symbolic/dynamic-shape compiler;
- multiprocessing/prefetch/pinned-memory DataLoader;
- NCCL/network DDP;
- production caching allocator;
- whole-Python compiler;
- general higher-order autodiff;
- PyTorch API/source/performance parity.

## Frozen-branch rule

Future major architecture does not belong on `v1.0`.

Allowed changes after freeze are limited to:

- documentation corrections;
- tests clarifying already-promised v1.0 behavior;
- narrow fixes inside the frozen v1.0 contract.

## After v1.0

There is intentionally no automatic `v1.1`.

Any future version requires a new, separately justified missing mental model and a new explicit roadmap starting from the latest validated `main`.
