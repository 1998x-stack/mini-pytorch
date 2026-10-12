# Agents.md — mini-pytorch Development Contract

> **Mandatory for every future iteration:** read this file before changing code, tests, documentation, branches, version scope, or architecture.
>
> This is the canonical engineering contract for `main`. Version branches carry branch-specific `Agents.md` files.

## 1. Repository mission

mini-pytorch is an educational, NumPy-first framework for understanding the mechanisms behind real PyTorch one architectural layer at a time.

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
v0.1  Tensor / Dynamic Autograd                         ✅
v0.2  Optimizer state / LR Scheduler                   ✅
v0.3  Dataset / Sampler / DataLoader / MNIST           ✅
v0.4  Storage / stride / views / version safety        ✅
v0.5  Operator Registry / Dispatcher / Redispatch      ✅
v0.6  Neural-network operator stack / CNN              ✅
v0.7  Device / Backend abstraction                     ✅
v0.8  Advanced Autograd Engine                         ✅
v0.9  Training Runtime / Serialization / Reproducibility ← next
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
                         Tensor / GradFn
                               │
                               ▼
                         Context graph
                               │
                               ▼
                         AutogradEngine
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
       local grad flow    graph lifetime      hooks
                         retain_graph
                               │
                               ▼
                     persistent requested grads

nn.Module → functional NN op ───────────────► Dispatcher
Dataset → Sampler → BatchSampler → DataLoader
Module / Parameter → Optimizer → LR Scheduler
```

`edu` is a deterministic simulated backend using host NumPy arrays. It exists to teach backend identity/routing and must never be represented as CUDA or GPU acceleration.

## 6. Architecture rules

- separate policy from mechanism;
- avoid circular core imports;
- backend kernels must not depend on Module/Optimizer/DataLoader/runtime checkpoint code;
- Autograd policy must not be duplicated inside backend kernels;
- preserve Storage/stride/view/version invariants;
- high-level `nn.Module` owns state/configuration, not backend loops;
- functional operators bridge Modules to logical operators;
- backend selection belongs in Device/Backend/Dispatcher, not scattered `if device` conditions;
- mixed-device operations fail explicitly unless the API is a transfer operation;
- Autograd Engine owns transient backward execution state; Tensor owns persistent user-visible grad/hook intent;
- graph lifetime, retained grads and hooks must remain backend-agnostic;
- v0.9 persistence/runtime code must not be embedded into Tensor numerical operators or Autograd Engine traversal;
- checkpoint loading must validate state instead of silently reshaping or dropping incompatible state.

## 7. Conceptual PyTorch compatibility

Prefer PyTorch-like boundaries when they improve transfer of understanding:

- TensorImpl-like Storage + sizes + strides + offset + device metadata;
- dynamic Function/GradFn/Context graph;
- explicit Autograd Engine execution/lifetime policy;
- leaf vs retained non-leaf gradients;
- Tensor gradient hooks;
- Module Parameters and Buffers;
- Optimizer parameter groups/state;
- Dataset/Sampler/DataLoader separation;
- operator schema + Dispatcher + DispatchKey;
- redispatch from policy wrappers to backend kernels;
- explicit Device + backend-owned allocation;
- next: complete training state + checkpoint/resume semantics.

Implement original educational equivalents.

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
v0.8
123 tests
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

## 11. Runtime/reproducibility rule for v0.9

The next version should treat training as a state machine rather than "model weights plus a loop".

State categories must be explicit:

```text
model state
optimizer state
scheduler state
RNG state
training progress
checkpoint schema/version metadata
```

Checkpoint serialization and state collection should be separate concerns.

Prefer transparent formats suitable for inspection; avoid opaque serialization when a small NPZ + JSON design is enough for the educational goal.

Deterministic-resume claims require an integration test comparing uninterrupted training against save→reload→continue execution.

## 12. Performance and dependency rule

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

Do not commit generated datasets, large logs, plots or checkpoints unless a tiny deterministic fixture is required by tests.

NumPy remains the reference numerical implementation. Future accelerator work must not make ordinary CPU CI impossible.

## 13. Definition of Done for a version

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

## 14. Current main directive

`main` contains the completed and validated **v0.8 Advanced Autograd Engine** milestone.

The next version must be created from current `main` and should be **v0.9: Training Runtime / Serialization / Reproducibility**.

Central question:

> What exact state must be captured, validated and restored so that a stopped training run can continue with the same parameters, optimizer dynamics, learning-rate schedule, random choices and progress counters?

v0.9 should build a checkpoint/runtime layer above existing Module/Optimizer/Scheduler/DataLoader mechanisms without pushing persistence logic into Tensor math or Autograd Engine execution.