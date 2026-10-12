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
v0.10 Distributed / DDP Concepts                         ← next
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
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
       local grad flow    graph lifetime       hooks
                               │
                               ▼
                    persistent requested grads

Dataset → Sampler → BatchSampler → DataLoader

Module / Parameter → Optimizer → LR Scheduler
                           │
                           ├──────────────┐
                           ▼              ▼
                    TrainingState   framework RNG
                           │              │
                           └──────┬───────┘
                                  ▼
                           CheckpointState
                                  ▼
                    metadata.json + tensors.npz
                                  ▼
                       validated restoration
```

`edu` is a deterministic simulated backend using host NumPy arrays. It teaches backend identity/routing and must never be represented as CUDA/GPU acceleration.

## 6. Architecture rules

- separate policy from mechanism;
- avoid circular core imports;
- backend kernels must not depend on Module/Optimizer/DataLoader/runtime/distributed layers;
- Autograd policy must not be duplicated inside backend kernels;
- preserve Storage/stride/view/version invariants;
- high-level `nn.Module` owns state/configuration, not backend loops;
- functional operators bridge Modules to logical operators;
- backend selection belongs in Device/Backend/Dispatcher, not scattered `if device` conditions;
- mixed-device operations fail explicitly unless the API is a transfer operation;
- AutogradEngine owns transient backward execution state; Tensor owns persistent user-visible grad/hook intent;
- checkpoint/runtime logic stays above Tensor/Dispatcher/AutogradEngine;
- distributed synchronization must stay separate from numerical kernels and optimizer algorithms;
- do not hide synchronization as arbitrary mutation inside an operator kernel.

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
- next: rank/world-size, ProcessGroup, collectives, DistributedSampler, DDP reducer and gradient buckets.

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
v0.9
140 tests
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

The v0.9 deterministic-resume test is a historical regression and must stay green while distributed work is added.

Do not silently expand v0.9 claims to arbitrary multi-worker mid-epoch restoration or private/external RNGs.

## 12. v0.10 distributed design rules

The purpose of v0.10 is to understand **gradient synchronization architecture**, not to implement production NCCL.

Prefer this layering:

```text
local AutogradEngine
       ↓
parameter gradient-ready hook
       ↓
DDP Reducer
       ↓
gradient buckets
       ↓
ProcessGroup collective
       ↓
averaged gradients
       ↓
Optimizer
```

Required rules:

- `rank` and `world_size` are explicit validated concepts;
- `ProcessGroup` owns collective semantics, not Module or Optimizer;
- `DistributedSampler` owns dataset index partitioning, not DataLoader internals;
- Reducer uses Autograd/Tensor hooks to observe readiness rather than modifying backward formulas;
- bucket flatten/unflatten must preserve parameter gradient shape/order;
- synchronization must happen before optimizer step;
- average-vs-sum semantics must be explicit;
- all replicas must use compatible parameter/bucket layouts;
- failures such as missing rank, wrong shape or collective-order mismatch must surface clearly.

### Simulation rule

v0.10 should work on ordinary single-process CPU CI.

A deterministic in-process collective simulation is preferred before multiprocessing/network transport.

Because sequential simulated ranks cannot block inside a hook waiting for peers, design synchronization as two phases where useful:

```text
backward hook
→ mark/stage gradient ready

explicit synchronization barrier
→ collective reduce staged buckets
→ write averaged grads back
```

Do not fake synchronous network semantics with deadlocking hooks.

## 13. Performance and dependency rule

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

NumPy remains the reference numerical implementation. Distributed tests must not require GPUs, NCCL, MPI or external services.

Do not commit generated datasets, logs, plots or checkpoints except tiny deterministic fixtures when strictly required.

## 14. Definition of Done for a version

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

## 15. Current main directive

`main` contains the completed and validated **v0.9 Training Runtime / Serialization / Reproducibility** milestone.

The next version must be created from current `main` and should be **v0.10: Distributed / DDP Concepts**.

Central question:

> If multiple model replicas run local forward/backward on different samples, how are gradients partitioned, marked ready, reduced and written back so every replica performs an equivalent optimizer step?

v0.10 should build the conceptual chain:

```text
DistributedSampler
      +
rank / world_size
      ↓
local backward
      ↓
grad-ready hooks
      ↓
Reducer / buckets
      ↓
ProcessGroup.all_reduce
      ↓
replica-consistent gradients
```

Real multiprocessing, networking and NCCL are explicit later concerns.
