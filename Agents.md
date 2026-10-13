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

Read:

1. `Agents.md`
2. `Plan.md`
3. `version.md`
4. relevant architecture docs
5. tests for the subsystem
6. current implementation

### On a version branch

Read:

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

Forbidden:

```text
version branch contains newer validated framework code
while main remains stale
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
v0.12 Guards / Graph Cache / Recompile                   ← next
```

One version should introduce one dominant mental model.

## 5. Current integrated architecture

```text
                             Tensor
                shape / stride / dtype / device
                               │
                               ▼
                             Storage
                  backend-owned allocation
                     + VersionCounter
                               │
                               ▼
                        logical operator
                               │
                               ▼
                           Dispatcher
                    policy keys + backend keys
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
          Autograd           NoGrad        CPU / EDU backend
             │                                   │
             └────────── redispatch ─────────────┘
                               │
                               ▼
                         numerical kernel
                               │
                               ▼
                         Tensor / GradFn
                               │
                               ▼
                         AutogradEngine

Dataset → Sampler → BatchSampler → DataLoader

Module / Parameter → Optimizer → LR Scheduler
                           │
                           ▼
                    Training Runtime
                  RNG + CheckpointState

DistributedSampler
      ↓
DDP → Reducer → GradientBucket → ProcessGroup

Python eager function
      ↓
Dispatcher.dispatch
      ├── normal execution
      └── capture observer
              ↓
       Value / Node / Graph
              ↓
       GraphInterpreter
              ↓
          Dispatcher replay
```

`edu` is a deterministic simulated backend using host NumPy arrays. It teaches backend identity/routing and must never be represented as CUDA/GPU acceleration.

## 6. Architecture rules

- separate policy from mechanism;
- avoid circular core imports;
- backend kernels must not depend on Module/Optimizer/DataLoader/runtime/distributed/compiler layers;
- Autograd policy must not be duplicated inside backend kernels;
- preserve Storage/stride/view/version invariants;
- high-level `nn.Module` owns state/configuration, not backend loops;
- functional operators bridge Modules to logical operators;
- backend selection belongs in Device/Backend/Dispatcher;
- AutogradEngine owns transient backward execution state;
- checkpoint/runtime logic stays above Tensor/Dispatcher/AutogradEngine;
- distributed synchronization stays separate from numerical kernels and optimizer algorithms;
- compiler capture records logical semantics and must not become a second backend/runtime;
- graph replay reuses Dispatcher rather than calling backend kernels directly;
- unsupported compiler dataflow fails explicitly rather than silently changing semantics.

## 7. Conceptual PyTorch compatibility

Prefer PyTorch-like boundaries when they improve transfer of understanding:

- TensorImpl-like Storage + sizes + strides + offset + device metadata;
- dynamic Function/GradFn/Context graph;
- explicit Autograd Engine;
- Module Parameters and Buffers;
- Optimizer parameter groups/state;
- Dataset/Sampler/DataLoader separation;
- operator schema + Dispatcher + DispatchKey;
- Device + backend-owned allocation;
- explicit training runtime/checkpoint state;
- rank/world-size, ProcessGroup, collectives, DDP reducer/buckets;
- logical graph representation separated from eager numerical execution;
- next: guards, specialization assumptions, graph cache and recompile.

Implement original educational equivalents rather than copying upstream internals.

## 8. Testing contract

Every new subsystem should include, where applicable:

- happy path;
- invariant/property tests;
- error-path tests;
- numerical/reference validation;
- integration test;
- regression coverage for earlier versions.

Full regression:

```bash
python -m unittest discover -s tests -v
```

Never claim a version is validated unless the exact reported code has passed the relevant suite. Before freeze, GitHub Actions must pass on the exact final version-branch head.

Current integrated baseline:

```text
v0.11
176 tests
OK
```

## 9. Documentation contract

When architecture/public API changes, review at minimum:

```text
version.md
README.md
Agents.md
MANIFEST.txt
relevant docs/*
relevant example
Plan.md on main
```

Every version README must state:

- version theme;
- delta from previous version;
- important source files;
- recommended reading order;
- quick run commands;
- explicit non-goals;
- next version.

## 10. Historical branch rule

A frozen version branch may receive only:

- documentation corrections;
- tests clarifying promised behavior;
- narrowly scoped bug fixes inside that version boundary.

Do not backport future subsystems.

## 11. Compiler invariant inherited from v0.11

Graph capture semantics are frozen as:

```text
outer Dispatcher.dispatch
      ↓
record one logical Node
```

not:

```text
Autograd wrapper
+ redispatch
+ backend kernel
= multiple semantic nodes
```

GraphInterpreter must continue to replay through Dispatcher.

GraphBreak remains explicit for unsupported dynamic non-leaf dataflow.

The v0.11 eager-vs-graph forward/backward equivalence tests are historical regressions and must remain green.

## 12. v0.12 guard/cache design rules

The purpose of v0.12 is to understand **graph validity and reuse**, not to implement Python frame evaluation or Dynamo itself.

Preferred layering:

```text
runtime call
    ↓
InputSignature
    ↓
GuardSet
    ↓
GraphCache lookup
 ┌──────┴──────┐
 ▼             ▼
hit           miss
 │              │
GraphInterpreter capture
 │              ↓
 │          new cache entry
 └──────────────┘
```

Required principles:

- guards describe assumptions, not numerical execution;
- cache keys must be deterministic and inspectable;
- shape/dtype/device/requires-grad/grad-mode distinctions must be explicit;
- constant identity/value policy must be documented;
- guard failure must be explainable;
- cache hit must replay the existing Graph through GraphInterpreter;
- cache miss must capture a new Graph rather than mutate an incompatible one;
- recompilation counters/statistics must be observable;
- no silent use of a graph whose assumptions no longer hold.

### Important boundary

v0.12 should not claim general dynamic-shape support.

A valid educational first model is specialization:

```text
shape + dtype + device + requires_grad + grad_mode
```

Different signatures may produce distinct cache entries.

Symbolic shape reasoning can come later.

## 13. Performance and dependency rule

Prefer:

```text
readable reference
   ↓
correctness
   ↓
inspectability
   ↓
profile
   ↓
optimized educational path
```

NumPy remains the reference numerical implementation.

Do not introduce heavy compiler dependencies for v0.12.

## 14. Definition of Done for a version

A version can be frozen only when:

- primary mental model is explicit;
- promised implementation is complete;
- unit/invariant/error tests pass;
- integration proof exists;
- educational example exists;
- `version.md` accurately records the delta;
- version-specific `Agents.md` exists;
- architecture docs are updated;
- package imports cleanly;
- exact final branch-head CI is green;
- validated state is integrated into `main`;
- next-version rationale is recorded in `Plan.md`.

## 15. Current main directive

`main` contains completed and validated **v0.11 Graph IR / Eager-to-Graph Capture**.

The next version must be created from current `main` and should be **v0.12: Guards / Graph Cache / Recompile**.

Central question:

> Once an eager program has been captured as Graph IR, what assumptions make that graph valid for a future call, and how should the runtime decide between cache hit and recapture?

Real Python bytecode/frame interception, symbolic shapes and production Dynamo behavior are explicit later concerns.
