# Agents.md — mini-pytorch Development Contract

> **Mandatory:** every future iteration reads this file before changing code, tests, docs, branches or architecture.
>
> This is the canonical contract for `main`. Frozen version branches carry branch-specific contracts.

## 1. Mission

mini-pytorch is a NumPy-first educational framework for understanding real PyTorch architectural boundaries one layer at a time.

Priority order:

```text
architectural clarity
      ↓
correctness / invariants
      ↓
inspectability
      ↓
reference proofs / tests
      ↓
measurement
      ↓
optimization
```

Do not optimize for API completeness and do not copy upstream PyTorch source.

## 2. Mandatory branch workflow

```text
validated main
   ↓
create vX.Y
   ↓
write branch version.md + Agents.md first
   ↓
implement one dominant mental model
   ↓
full tests + docs + example
   ↓
exact branch-head CI green
   ↓
freeze historical snapshot
   ↓
exact frozen-head CI green
   ↓
fast-forward validated tree into main
   ↓
restore canonical README / Agents / Plan
   ↓
exact main-head CI green
```

`main` must remain the newest integrated validated framework state.

## 3. Version sequence

```text
v0.1  Tensor / Dynamic Autograd                           ✅
v0.2  Optimizer state / LR Scheduler                     ✅
v0.3  Dataset / Sampler / DataLoader / MNIST             ✅
v0.4  Storage / stride / views / version safety          ✅
v0.5  Operator Registry / Dispatcher / Redispatch        ✅
v0.6  Neural-network operator stack / CNN                ✅
v0.7  Device / Backend abstraction                       ✅
v0.8  Advanced Autograd Engine                           ✅
v0.9  Training Runtime / Serialization / Reproducibility ✅
v0.10 Distributed / DDP Concepts                         ✅
v0.11 Graph IR / Eager-to-Graph Capture                  ✅
v0.12 Guards / Graph Cache / Recompile                   ✅
v0.13 AOTAutograd-like Graph Differentiation             ✅
v0.14 Lowering / Fusion / Inductor-like Concepts         ✅
v0.15 Runtime / Performance Mechanics                    ← next
```

## 4. Current integrated compiler stack

```text
Python eager
   ↓
Dispatcher capture
   ↓
Graph IR
   ↓
Guards / GraphCache
   ↓
AOT differentiation
   ├── forward Graph
   └── backward Graph
         ↑
 saved Values + grad_output
   ↓
normalize / analyze
   ↓
DCE / constant folding
   ↓
use-aware fusion
   ↓
LoweredProgram
   ├── ExternalOp
   └── PointwiseFusion
   ↓
LoweredExecutor
```

Frozen compiler invariants through v0.14:

- capture records outer logical Dispatcher operations, not redispatch internals;
- Graph IR does not bake in backend/Autograd dispatch decisions;
- guards/cache specialize rather than mutate incompatible Graphs;
- AOT backward uses explicit saved Values and explicit gradient accumulation;
- source Graphs are not mutated by v0.14 passes;
- use-count/liveness data is explicit;
- DCE operates only under the documented pure-Graph assumption;
- grad-requiring constants are not folded;
- pointwise fusion is limited to `add/mul/relu` in v0.14;
- branching/shared values stop the conservative linear-chain fusion model;
- matmul/reduction/conv/pool/AOT helpers remain fusion barriers;
- LoweredExecutor is first-order and runs under `no_grad`;
- source-to-lowered Value mapping preserves AOT saved-value ABI;
- reduced instruction count is not a performance claim.

## 5. Cross-layer architecture rules

- separate policy from mechanism;
- avoid circular core imports;
- backend kernels stay independent of Module/Optimizer/DataLoader/runtime/distributed/compiler policy;
- Autograd policy stays out of numerical kernels;
- preserve Storage/stride/view/version invariants;
- backend selection belongs to Device/Dispatcher;
- AutogradEngine owns eager backward runtime state;
- checkpoint/runtime logic stays above Tensor/Dispatcher;
- distributed synchronization stays outside numerical kernels and optimizer algorithms;
- compiler transformations do not become a second Tensor/backend architecture;
- unsupported behavior fails explicitly rather than silently changing semantics.

## 6. Testing contract

Every subsystem should include, where applicable:

- happy path;
- invariant/property tests;
- error-path tests;
- numerical/reference validation;
- integration proof;
- full historical regression.

Run:

```bash
python -m unittest discover -s tests -v
```

Current integrated baseline:

```text
v0.14
221 tests
OK
```

Never claim a version or performance result unless the exact reported state was actually measured/validated.

## 7. Documentation contract

Review on each version:

```text
README.md
version.md
Agents.md
MANIFEST.txt
relevant docs/*
relevant example
Plan.md on main
```

Every version README/version file must explain the delta, architecture, invariants, tests, non-goals, reading order and next dependency.

## 8. Frozen branch rule

Frozen version branches may receive only:

- documentation corrections;
- tests clarifying already-promised behavior;
- narrowly scoped fixes inside that historical version boundary.

Future architecture belongs on a new version branch.

## 9. v0.15 design rules — Runtime / Performance Mechanics

Central question:

> Once LoweredProgram is correct, how should runtime execution be measured and how can memory/work scheduling be improved using explicit evidence?

Target layering:

```text
LoweredProgram
      ↓
RuntimeProfiler
      ↓
Value lifetime analysis
      ↓
BufferPlanner
      ↓
ExecutionPlan
      ↓
PlannedExecutor
      ↓
BenchmarkReport
```

### Measurement before optimization

No optimization may be justified by intuition alone.

Benchmark protocol must define:

```text
warmup
repeat count
measured iterations
median / distribution
input shapes/dtypes/device
executor variant
```

Do not advertise a speedup from a single timing sample.

### Profiling

Runtime profiling should expose per-instruction data such as:

```text
instruction id/type
operator/fusion name
elapsed time
output shapes
allocation events
```

Instrumentation must be optional and must not change numerical semantics.

### Value lifetime

Derive explicit intervals from `LoweredProgram`:

```text
birth instruction
last-use instruction
is graph input
is constant
is graph output
```

Do not reuse a value's storage before its last use.

### Buffer planning

Start conservatively.

A buffer may be reused only when compatibility is explicit, for example:

```text
lifetimes do not overlap
shape/numel compatible
dtype identical
device identical
not externally owned input/constant/output
```

Buffer reuse must be represented as plan data, not hidden mutation.

### ExecutionPlan

Expected conceptual representation:

```text
ExecutionPlan
├── instructions
├── value lifetimes
├── value → buffer slot assignments
├── allocation metadata
└── profiling/benchmark metadata
```

### Correctness oracle

For every optimization path:

```text
GraphInterpreter
      ==
LoweredExecutor
      ==
PlannedExecutor
```

AOT forward and backward paths remain mandatory integration tests.

## 10. v0.15 performance-reporting rules

- separate compile/planning time from steady-state execution time;
- report warmup policy;
- use repeated measurements;
- prefer median and distribution/range rather than one timing;
- compare identical inputs/workloads;
- distinguish CPU timing from EDU logical-device identity tests;
- report allocation/peak-live metrics separately from wall-clock speed;
- a slower result is valid evidence and must not be hidden;
- do not claim PyTorch performance parity.

## 11. v0.15 explicit non-goals

- CUDA/Triton code generation;
- GPU kernel benchmarking;
- production allocator;
- asynchronous stream scheduling;
- real multithreaded task scheduler;
- symbolic-shape memory planning;
- production-grade profiler;
- cherry-picked benchmark claims.

## 12. Definition of Done

A future version freezes only when:

- its dominant mental model is explicit;
- implementation and failure boundaries are complete;
- correctness/reference proofs are green;
- docs/example/manifest are complete;
- benchmark methodology is reproducible when performance is discussed;
- exact final version-head CI is green;
- validated tree is integrated into main;
- Plan.md advances the next architectural dependency.

## 13. Current directive

`main` contains completed **v0.14 Lowering / Fusion / Inductor-like Concepts**.

Next branch: **v0.15 — Runtime / Performance Mechanics**.
