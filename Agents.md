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
implement one major mental model
   ↓
full tests + docs + example
   ↓
exact final branch-head CI green
   ↓
freeze version branch
   ↓
integrate validated tree into main
   ↓
restore canonical main docs
   ↓
advance Plan.md
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
v0.11 Graph IR / Eager-to-Graph Capture                  ← next
```

One version should introduce one dominant mental model.

## 5. Current integrated architecture

```text
                              Tensor
                   shape / stride / dtype / device
                                │
                                ▼
                              Storage
                 backend-owned allocation + VersionCounter
                                │
                                ▼
                         logical operator
                                │
                                ▼
                            Dispatcher
               policy keys + backend dispatch keys
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
              Autograd        NoGrad        CPU / EDU
                 │                              │
                 └──────── redispatch ──────────┘
                                │
                                ▼
                         numerical kernel
                                │
                                ▼
                          Tensor / GradFn
                                │
                                ▼
                          AutogradEngine
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
           local grad flow  graph lifetime    hooks

Dataset → Sampler → BatchSampler → DataLoader
           │
           └──────────────→ DistributedSampler

Module / Parameter → Optimizer → LR Scheduler
        │
        ├───────────────┐
        ▼               ▼
 TrainingState      framework RNG
        │               │
        └───────┬───────┘
                ▼
         CheckpointState
                ▼
     metadata.json + tensors.npz
                ▼
       validated restoration

Replicated Modules
        ↓
leaf Parameter hooks
        ↓
      Reducer
        ↓
GradientBucket(s)
        ↓
    ProcessGroup
        ↓
     all_reduce
        ↓
synchronized gradients
```

`edu` is a deterministic simulated backend using host NumPy arrays. It teaches backend identity/routing and must never be represented as CUDA/GPU acceleration.

The v0.10 ProcessGroup world is also a deterministic in-process simulation. It teaches collective ordering and DDP architecture; it is not NCCL/Gloo/network execution.

## 6. Architecture rules

- separate policy from mechanism;
- avoid circular core imports;
- backend kernels must not depend on Module/Optimizer/DataLoader/runtime/distributed/compiler layers;
- Autograd policy must not be duplicated inside backend kernels;
- preserve Storage/stride/view/version invariants;
- high-level `nn.Module` owns state/configuration, not backend loops;
- functional operators bridge Modules to logical operators;
- backend selection belongs in Device/Backend/Dispatcher, not scattered `if device` conditions;
- mixed-device operations fail explicitly unless the API is a transfer operation;
- AutogradEngine owns transient backward execution state; Tensor owns persistent user-visible grad/hook intent;
- checkpoint/runtime logic stays above Tensor/Dispatcher/AutogradEngine;
- distributed synchronization stays separate from numerical kernels and optimizer algorithms;
- compiler/capture work must observe operator execution through a clear boundary rather than duplicating operator semantics;
- future graph IR must replay through existing Dispatcher/operator semantics where practical.

## 7. Conceptual PyTorch compatibility

Prefer PyTorch-like boundaries when they improve transfer of understanding:

- TensorImpl-like Storage + sizes + strides + offset + device metadata;
- dynamic Function/GradFn/Context graph;
- explicit Autograd Engine execution/lifetime policy;
- Tensor gradient hooks;
- Module Parameters and Buffers;
- Optimizer parameter groups/state;
- Dataset/Sampler/DataLoader separation;
- operator schema + Dispatcher + DispatchKey;
- Device + backend-owned allocation;
- explicit training runtime/checkpoint state;
- rank/world-size + ProcessGroup + DistributedSampler + DDP Reducer/buckets;
- next: explicit `Graph / Node / Value` IR captured from eager operator execution.

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
v0.10
159 tests
OK
```

## 9. Documentation contract

When architecture/public API changes, review at minimum:

```text
version.md
README.md
Agents.md
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

## 11. Runtime/checkpoint invariant inherited from v0.9

Training resume state remains explicit:

```text
model state
optimizer state
scheduler state
framework RNG state
TrainingState
schema metadata
```

The deterministic-resume integration test remains a permanent regression.

## 12. Distributed invariant inherited from v0.10

The distributed teaching model remains:

```text
DistributedSampler
      ↓
rank-local backward
      ↓
Parameter hooks
      ↓
Reducer / buckets
      ↓
ProcessGroup all_reduce
      ↓
synchronized gradients
```

Preserve these rules:

- rank/world-size explicit;
- collective sequence/op/shape/dtype compatibility explicit;
- no blocking hook waits in sequential simulation;
- Reducer owns bucket flatten/unflatten;
- optimizer remains distribution-unaware;
- global-batch reference-equivalence test remains green;
- EDU placement remains unchanged by synchronization.

## 13. v0.11 Graph IR design rules

The purpose of v0.11 is to understand how eager execution becomes explicit graph data, not to build a production compiler.

Preferred layering:

```text
Tensor eager API
      ↓
Dispatcher operator call
      ↓
Capture observer/context
      ↓
Graph / Node / Value
      ↓
GraphInterpreter
      ↓
existing Dispatcher
```

Required rules:

- IR objects are pure, inspectable metadata/data structures;
- graph capture must not replace or fork numerical operator implementations;
- eager execution remains the source of truth during tracing;
- graph replay should invoke existing logical operators / Dispatcher registrations;
- graph inputs, constants, node outputs and graph outputs must have explicit identities;
- positional args, kwargs and operator schema names must be represented deterministically;
- unsupported capture must produce an explicit graph break/error boundary rather than silently dropping operations;
- graph printing should be stable and readable;
- v0.11 does not add guard caches, fusion or lowering optimizations.

### Capture model

Prefer an educational eager-tracing approach first:

```text
real Tensor inputs
      ↓
execute eager op normally
      ↓
record logical operator + input Value ids + attributes
      ↓
map real output Tensor identity → output Value id
```

The captured graph should be replayable with new compatible input Tensors.

### Autograd boundary

GraphInterpreter may replay existing Tensor operators, so replayed outputs can naturally build a fresh eager Autograd graph.

Do not serialize/reuse old GradFn objects inside IR.

### Graph-break boundary

Operations not routed through the capture-supported operator path must fail or break explicitly. Do not claim whole-program capture for Python control flow, arbitrary NumPy calls or unsupported direct `Function.apply` paths.

## 14. Performance and dependency rule

Prefer:

```text
readable reference
   ↓
correctness
   ↓
profile
   ↓
optimized educational path
```

NumPy remains the reference numerical implementation. Compiler/capture tests must not require external compiler packages.

Do not commit generated datasets, logs, plots or checkpoints except tiny deterministic fixtures when strictly required.

## 15. Definition of Done for a version

A version can be frozen only when:

- primary mental model is explicit;
- promised implementation is complete;
- unit/invariant/error tests pass;
- numerical/reference checks exist where relevant;
- educational integration example exists;
- `version.md` accurately records the delta;
- version-specific `Agents.md` exists;
- architecture docs are updated;
- package imports cleanly;
- exact final branch-head CI is green;
- validated state is integrated into `main`;
- next-version rationale is recorded in `Plan.md`.

## 16. Current main directive

`main` contains the completed and validated **v0.10 Distributed / DDP Concepts** milestone.

The next version must be created from current `main` and should be **v0.11: Graph IR / Eager-to-Graph Capture**.

Central question:

> How can a sequence of eager operator calls be represented as explicit `Value / Node / Graph` data, then replayed through the existing Dispatcher to produce equivalent results?

v0.11 should build:

```text
capture context
      +
Value / Node / Graph
      ↓
supported-op recording
      ↓
inputs / constants / outputs
      ↓
GraphInterpreter
      ↓
eager-vs-graph forward/backward equivalence
```

Guard specialization, graph caches and recompilation belong to v0.12, not v0.11.
