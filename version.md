# mini-pytorch v1.0

## Version identity

- **Branch:** `v1.0`
- **Previous version:** `v0.15`
- **Theme:** Integrated Architecture
- **Status:** release candidate
- **Package version:** `1.0.0`
- **Starting validated baseline:** 247 tests / OK
- **Current release-candidate baseline:** **251 / 251 tests passed**
- **Base main commit:** `56191655f1188134e62e965dc58fdd3dcd3d062b`
- **Validated framework/CI commit:** `d49b1872274f790303d719034074b75c270a18c2`

v1.0 answers:

> Can the complete mini-pytorch learning stack be traced, explained, exercised and validated as one coherent architecture without adding another major subsystem?

The release-candidate answer is yes: the 0.x mechanism-building sequence now composes through explicit cross-layer tests and one end-to-end executable example.

## Core delta from v0.15

v0.15 completed the last new mechanism:

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

v1.0 consolidates instead of extending:

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
release-quality validation
```

## Completed deliverables

### Architecture audit

Added:

```text
docs/ARCHITECTURE_V1.md
```

It audits the actual source ownership/dependency/failure boundaries for:

```text
Device / Backend / Storage / Tensor
Dispatcher / kernels
AutogradEngine
Module / Optimizer / Data
Runtime / Checkpoint / RNG
Distributed / DDP
Capture / Guards / Cache
AOT differentiation
Lowering / Fusion
Lifetime / MemoryPlan / ExecutionPlan
PlannedExecutor
```

### Public API / naming audit

Added:

```text
docs/API_AUDIT.md
```

Result: no breaking API rewrite is justified. Low-level classes remain intentionally public learning surfaces.

Small v1.0 cleanup:

```text
mini_pytorch/_version.py
      ↓
mini_pytorch.__version__
checkpoint framework_version
pyproject package version
```

all use release `1.0.0` semantics.

### Cross-layer tests

Added:

```text
tests/test_v1_integration.py
```

It proves composition across:

```text
Tensor → Dispatcher → Backend → Autograd
Module → Optimizer → Scheduler → Checkpoint restore
Capture → Guards/Cache → AOT → Lowering → ExecutionPlan
Planned backward → eager gradient reference
```

### Integrated example

Added:

```text
examples/19_integrated_architecture.py
```

The example separates real phases:

```text
1. eager training
2. runtime/checkpoint state
3. guarded graph cache
4. eager gradient reference
5. AOT + lowering + planned runtime
```

GitHub Actions now runs this example as a release gate.

### PyTorch conceptual mapping

Added:

```text
docs/PYTORCH_MAPPING.md
```

The map explicitly distinguishes:

```text
conceptual responsibility correspondence
!=
source/API/performance parity
```

### Documentation reconciliation

Updated/replaced stale canonical docs:

```text
docs/ARCHITECTURE.md
docs/LEARNING_PATH.md
docs/ROADMAP.md
MANIFEST.txt
README.md
```

The old roadmap that still described v0.3 as active has been replaced by the completed learning sequence.

### CI governance

The old workflow only matched:

```text
main
v0.*
```

which excluded `v1.0`.

v1.0 changes the branch gate to:

```text
main
v*
```

and adds:

```bash
python examples/19_integrated_architecture.py
```

after the full unit suite.

## Integrated architecture

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

## Preserved invariants

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

## Correctness proofs

For supported execution:

```text
Eager Tensor execution
      ==
GraphInterpreter
      ==
LoweredExecutor
      ==
PlannedExecutor
```

For the supported AOT integration path:

```text
eager Autograd gradients
      ==
AOT explicit backward Graph
      ==
lowered/planned backward gradients
```

## Release-candidate validation

Validated commit:

```text
d49b1872274f790303d719034074b75c270a18c2
```

GitHub Actions environment:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

Unit-test result:

```text
Ran 251 tests in 1.154s
OK
```

Integrated-example result includes:

```text
mini-pytorch 1.0.0
cache_entries=1
compile_count=1
hit_count=1
planned output matches eager: PASS
planned gradients match eager: PASS
```

## Important v1.0 files

```text
mini_pytorch/_version.py
tests/test_v1_integration.py
examples/19_integrated_architecture.py
docs/ARCHITECTURE_V1.md
docs/API_AUDIT.md
docs/PYTORCH_MAPPING.md
```

## Explicit non-goals

v1.0 does not add:

- real CUDA/Triton code generation;
- a new accelerator backend;
- symbolic dynamic-shape compiler;
- production multiprocessing DataLoader;
- NCCL/network DDP;
- production allocator;
- whole-Python compiler;
- complete higher-order autodiff;
- PyTorch API feature parity;
- marketing-style benchmark claims.

## Remaining release steps

No feature work remains for v1.0.

The only remaining steps are:

```text
exact release-candidate docs HEAD CI
      ↓
freeze version.md + Agents.md
      ↓
exact frozen HEAD CI
      ↓
fast-forward into main
      ↓
restore canonical integrated main docs
      ↓
exact main HEAD CI
```

## After v1.0

There is no automatic `v1.1` roadmap. Any post-v1.0 expansion requires a separately justified missing mental model and a new explicit roadmap.
