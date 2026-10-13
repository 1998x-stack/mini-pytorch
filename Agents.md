# Agents.md — mini-pytorch Development Contract

> **Mandatory:** every future iteration reads this file before changing code, tests, docs, branches or architecture.
>
> This is the canonical contract for `main`. Frozen version branches carry branch-specific contracts.

## 1. Mission

mini-pytorch is a small NumPy-first framework for learning real PyTorch architectural boundaries.

Priority order:

```text
architectural clarity
      ↓
correctness / invariants
      ↓
inspectability
      ↓
tests / reference proofs
      ↓
PyTorch conceptual transfer
      ↓
performance
```

Do not optimize for API completeness and do not copy upstream PyTorch source.

## 2. Branch workflow

```text
validated main
   ↓
create vX.Y
   ↓
branch version.md + Agents.md
   ↓
implement one dominant mental model
   ↓
tests + docs + example
   ↓
exact branch-head CI green
   ↓
freeze branch
   ↓
fast-forward validated tree into main
   ↓
restore canonical README / Agents / Plan
   ↓
advance next milestone
```

`main` must never intentionally remain behind a newer validated version branch.

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
v0.14 Lowering / Fusion / Inductor-like Concepts         ← next
```

## 4. Current integrated compiler stack

```text
Python eager
   ↓
Dispatcher-level capture
   ↓
Graph IR
   ↓
Guards / GraphCache
   ↓
GraphInterpreter
   ↓
AOT differentiation
   ├── forward Graph
   └── backward Graph
         ↑
 saved Values + grad_output
```

Frozen invariants:

- capture records logical outer dispatch, not redispatch internals;
- Graph IR does not bake in backend/Autograd decisions;
- GraphInterpreter executes through Dispatcher;
- guard failures are normal inspectable cache misses;
- cache misses create new specializations rather than mutate incompatible graphs;
- GraphBreak never installs partial cache state;
- AOT backward uses explicit saved Values and explicit gradient accumulation;
- eager AutogradEngine remains the v0.13 reference oracle;
- v0.13 backward Graph execution is first-order; no higher-order claim.

## 5. Cross-layer architecture rules

- separate policy from mechanism;
- avoid circular core imports;
- backend kernels do not depend on Module/Optimizer/DataLoader/runtime/distributed/compiler policy;
- Autograd policy stays out of numerical kernels;
- preserve Storage/stride/view/version invariants;
- backend selection belongs to Device/Dispatcher;
- AutogradEngine owns eager backward runtime state;
- training/checkpoint logic remains above Tensor/Dispatcher;
- distributed synchronization remains separate from optimizer algorithms;
- compiler layers transform/execute logical programs, not create a second tensor backend;
- unsupported compiler behavior fails explicitly.

## 6. Testing contract

Every new subsystem needs, where applicable:

- happy path;
- invariant/property tests;
- error-path tests;
- numerical/reference validation;
- integration proof;
- full historical regression.

Command:

```bash
python -m unittest discover -s tests -v
```

Current integrated baseline:

```text
v0.13
206 tests
OK
```

Never claim validation unless the exact reported HEAD passed CI.

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

Each version branch must clearly state its delta from the prior version and its non-goals.

## 8. Frozen branch rule

Frozen version branches may receive only:

- documentation corrections;
- tests clarifying promised behavior;
- narrowly scoped fixes inside that version boundary.

Future subsystems belong on a new version branch.

## 9. v0.14 design rules — Lowering / Fusion

Central question:

> Given explicit forward/backward Graph programs, how can compiler transformations produce a simpler/lower-level execution plan while preserving semantics?

Target layering:

```text
Graph
  ↓
analysis
  ↓
normalization
  ↓
DCE / safe constant folding
  ↓
fusion grouping
  ↓
LoweredProgram
  ↓
reference executor
```

Required principles:

- transformation passes do not mutate the original historical Graph in-place unless explicitly designed/tested;
- every pass has before/after inspectability;
- use/liveness information is explicit;
- dead-value elimination must preserve graph outputs and side-effect assumptions;
- constant folding is restricted to provably safe pure operations;
- fusion groups must preserve dependency ordering;
- lowered execution must have eager/Graph reference equivalence tests;
- both forward and backward Graphs should be usable as lowering inputs;
- performance claims require measurement; reduced node count is not automatically faster;
- no Triton/CUDA-codegen claim.

### Recommended first fusion scope

Start with pure elementwise chains:

```text
add
mul
relu
```

Treat reductions and matmul as fusion boundaries initially.

## 10. v0.14 explicit non-goals

- production TorchInductor;
- Triton code generation;
- GPU scheduling;
- symbolic-shape compiler;
- arbitrary alias/mutation optimization;
- whole-program Python compilation;
- automatic kernel autotuning.

## 11. Definition of Done

A version freezes only when:

- its dominant mental model is explicit;
- implementation and error boundaries are complete;
- integration/reference proofs pass;
- docs/example/manifest are updated;
- exact final branch HEAD CI is green;
- validated tree is integrated into main;
- Plan.md explains the next dependency step.

## 12. Current directive

`main` contains completed **v0.13 AOTAutograd-like Graph Differentiation**.

Next branch: **v0.14 — Lowering / Fusion / Inductor-like Concepts**.
