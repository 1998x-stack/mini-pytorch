# Agents.md — mini-pytorch Development Contract

> **Mandatory for every future iteration:** read this file before changing code, tests, documentation, branches, version scope, or architecture.
>
> This is the canonical engineering contract for `main`. Version branches carry branch-specific `Agents.md` files.

## 1. Repository mission

mini-pytorch is an educational, NumPy-first framework for understanding real PyTorch mechanisms one architectural layer at a time.

Priorities:

```text
architectural clarity
      ↓
correctness + explicit invariants
      ↓
readable reference implementation
      ↓
tests that demonstrate mechanisms
      ↓
conceptual mapping to PyTorch
      ↓
performance only after the model is clear
```

Do not optimize for API completeness and do not copy upstream PyTorch source.

## 2. Mandatory reading order

### On `main`

1. `Agents.md`
2. `Plan.md`
3. `version.md`
4. relevant architecture docs
5. tests for the subsystem
6. current implementation

### On a version branch

1. branch `Agents.md`
2. branch `version.md`
3. relevant docs/tests/source

Never assume a historical branch should contain later-version functionality.

## 3. Branch invariant

`main` must always be the newest integrated, validated state.

```text
latest main
   ↓
create vX.Y
   ↓
write branch version.md + Agents.md first
   ↓
implement one dominant mental model
   ↓
full tests + docs + example
   ↓
exact final branch-head CI green
   ↓
freeze version branch
   ↓
fast-forward/integrate validated tree into main
   ↓
restore canonical main README / Agents / Plan
   ↓
advance next milestone
```

## 4. Version sequence

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
v0.13 AOTAutograd-like Graph Differentiation             ← next
```

One version should introduce one dominant mental model.

## 5. Current compiler invariants

The completed compiler stack is:

```text
Python eager call
      ↓
Dispatcher.dispatch
      ↓
Graph capture
      ↓
Value / Node / Graph
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
GraphInterpreter capture
 │              ↓
 └──────── Dispatcher replay
```

Frozen invariants inherited from v0.11/v0.12:

- capture records outer logical Dispatcher ops, not redispatch/backend internals;
- Graph IR does not bake in CPU/EDU/Autograd decisions;
- GraphInterpreter replays through Dispatcher;
- guard failures are inspectable normal cache misses;
- cache entries are separate specializations, not mutated incompatible graphs;
- exact v0.12 specialization fields are input count/shape/dtype/device/requires_grad/grad mode;
- GraphBreak never installs a partial cache entry;
- closure Tensor constants are capture-by-value snapshots in v0.12;
- symbolic shapes are not yet supported.

## 6. Architecture rules

- separate policy from mechanism;
- avoid circular core imports;
- backend kernels must not depend on Module/Optimizer/DataLoader/runtime/distributed/compiler layers;
- Autograd policy must not be duplicated inside backend kernels;
- preserve Storage/stride/view/version invariants;
- high-level `nn.Module` owns state/configuration, not backend loops;
- backend selection belongs in Device/Backend/Dispatcher;
- AutogradEngine owns transient eager backward execution state;
- checkpoint/runtime logic stays above Tensor/Dispatcher/AutogradEngine;
- distributed synchronization stays separate from numerical kernels and optimizer algorithms;
- compiler capture/cache must not become a second numerical backend;
- graph replay reuses Dispatcher;
- unsupported compiler dataflow fails explicitly.

## 7. Testing contract

Every new subsystem should include:

- happy path;
- invariant/property tests;
- error-path tests;
- numerical/reference validation;
- integration proof;
- regression coverage for earlier versions.

Full regression:

```bash
python -m unittest discover -s tests -v
```

Current integrated baseline:

```text
v0.12
192 tests
OK
```

Never claim a version is validated unless the exact final branch head passed CI.

## 8. Documentation contract

When architecture/public API changes, review:

```text
version.md
README.md
Agents.md
MANIFEST.txt
relevant docs/*
relevant example
Plan.md on main
```

Every version README must state theme, delta, important files, reading order, run commands, non-goals, and next version.

## 9. Historical branch rule

Frozen version branches may receive only documentation corrections, tests clarifying promised behavior, or narrow bug fixes inside that version boundary.

Do not backport future compiler stages.

## 10. v0.13 design rules — AOTAutograd-like

The purpose of v0.13 is to understand **backward as graph data**.

Target mental model:

```text
captured forward Graph
      ↓
identify differentiable outputs/inputs
      ↓
forward values needed by backward
      ↓
construct backward Graph
      ↓
seed grad outputs
      ↓
GraphInterpreter
      ↓
input gradients
```

Required principles:

- do not replace the eager AutogradEngine; keep it as correctness reference;
- backward graph construction must be explicit and inspectable;
- saved forward values must be explicit graph interface/state, not hidden Python closures;
- backward formulas should be expressed with graph/Tensor operators where possible;
- compare AOT-style gradients against eager AutogradEngine gradients;
- preserve Dispatcher/backend/device semantics;
- CPU and EDU behavior must remain consistent;
- do not claim higher-order gradients unless backward graph itself is differentiable and tested;
- do not begin lowering/fusion in v0.13.

### Recommended scope

Start with a deliberately small differentiable op set:

```text
add
mul
matmul
relu
sum
```

Then expand only if the graph-level differentiation model is stable.

### Explicit non-goals

v0.13 is not:

- production AOTAutograd;
- mutation functionalization for arbitrary Python programs;
- decompositions for all operators;
- symbolic shape system;
- Inductor/Triton lowering;
- general higher-order autodiff.

## 11. Definition of Done

A future version can freeze only when:

- primary mental model is explicit;
- promised implementation complete;
- tests/invariants/errors green;
- integration proof exists;
- educational example exists;
- `version.md` and version-specific `Agents.md` accurate;
- docs/manifest updated;
- exact final branch-head CI green;
- validated tree integrated into `main`;
- next-version rationale recorded in `Plan.md`.

## 12. Current main directive

`main` contains completed and validated **v0.12 Guards / Graph Cache / Recompile**.

The next branch must be created from current `main` as **v0.13 — AOTAutograd-like Graph Differentiation**.

Central question:

> Once a guarded forward Graph is reusable, how can reverse-mode differentiation itself become an explicit graph program rather than opaque eager VJP execution?
