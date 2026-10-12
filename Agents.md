# Agents.md — mini-pytorch Development Contract

> **Mandatory for every future iteration:** read this file before changing code, tests, documentation, branches, version scope, or architecture.
>
> This is the canonical engineering contract for `main`. Version branches carry their own branch-specific `Agents.md`.

## 1. Repository mission

mini-pytorch is an educational, NumPy-first framework for understanding the mechanisms behind real PyTorch.

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

The project does not optimize for API completeness or for copying upstream PyTorch source code.

## 2. Mandatory reading order

### On `main`

Read:

1. `Agents.md`
2. `Plan.md`
3. `version.md`
4. architecture docs relevant to the subsystem
5. tests for that subsystem
6. current implementation

### On a version branch

Read:

1. branch `Agents.md`
2. branch `version.md`
3. relevant docs/tests/source

Never assume a historical branch should contain a later-version feature.

## 3. Branch invariant

`main` must always be the newest integrated, validated state.

Correct lifecycle:

```text
latest main
   ↓
create vX.Y
   ↓
implement one major mental model
   ↓
full tests + docs + example
   ↓
exact branch-head CI green
   ↓
freeze version branch
   ↓
integrate validated tree to main
   ↓
restore canonical main docs
```

Forbidden:

```text
active/frozen version contains newer validated framework code
main remains stale
```

## 4. One version = one major mental model

Current learning sequence:

```text
v0.1  Tensor / Dynamic Autograd
v0.2  Optimizer state / LR Scheduler
v0.3  Dataset / Sampler / DataLoader / MNIST
v0.4  Storage / stride / views / inplace version safety
v0.5  Operator Registry / Dispatcher / Redispatch
v0.6  Neural-network operator stack / CNN        ← next
```

Features that do not serve the current version's mental model should be deferred to `Plan.md`.

## 5. Architecture rules

Preferred dependency direction:

```text
Tensor / Storage
      │
      ▼
Operator Schema / Dispatcher
      │
 ┌────┴───────────┐
 ▼                ▼
Backend kernels  Autograd wrappers
      │                │
      └──────┬─────────┘
             ▼
            nn
             │
             ▼
           optim

Dataset / Sampler / DataLoader
             │
             └──────────► training runtime
```

Rules:

- separate policy from mechanism;
- avoid circular core imports;
- CPU/backend kernels should not know Module/Optimizer/DataLoader;
- Autograd policy should not be duplicated inside every backend kernel;
- preserve Storage/stride/view invariants while extending operators;
- do not scatter device-specific `if` statements through Tensor before the device/backend version.

## 6. Conceptual PyTorch compatibility

Prefer PyTorch-like boundaries when they improve transfer of understanding:

- TensorImpl-like Storage + sizes + strides + storage offset;
- dynamic Autograd Functions/Nodes;
- Module parameters and buffers;
- Optimizer parameter groups/state;
- Dataset/Sampler/DataLoader separation;
- operator schema + Dispatcher + DispatchKey;
- redispatch from wrappers to backend kernels;
- later: Conv2d, device/backend, DDP and compiler stack.

Implement original educational equivalents. Do not copy upstream source.

## 7. Testing contract

Every new subsystem should include, as applicable:

- happy path;
- invariant/property tests;
- error-path tests;
- numerical/reference validation;
- integration test.

Full regression command:

```bash
python -m unittest discover -s tests -v
```

Never claim a version is validated unless the exact code being reported has actually passed the relevant test suite. Before freeze, GitHub Actions must pass on the exact final version-branch head.

## 8. Documentation contract

When architecture or public API changes, review at minimum:

```text
version.md
README.md
Agents.md
relevant docs/*
relevant example
Plan.md on main when version sequencing changes
```

Every version README must state:

- version theme;
- delta from previous version;
- important source files;
- recommended reading order;
- quick run commands;
- explicit non-goals;
- next version.

## 9. Historical branch rule

A frozen version branch may receive only:

- documentation corrections;
- tests that correct the version's own promised behavior;
- narrowly scoped bug fixes within that version's scope.

Do not backport future subsystems merely for convenience.

## 10. Performance rule

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

Examples:

- readable Conv2d loops before im2col optimization;
- eager kernels before fusion;
- simple synchronization before DDP buckets.

## 11. Generated artifacts and dependencies

Do not commit generated datasets, logs, plots or checkpoints unless a tiny deterministic fixture is explicitly required by tests.

NumPy remains the default reference backend until the device/backend version explicitly introduces another backend.

## 12. Definition of Done for a version

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
- next version dependency rationale is recorded in `Plan.md`.

## 13. Current main directive

`main` currently contains the completed v0.5 Dispatcher milestone.

The next branch must be created from current `main` and follow `Plan.md`. The next planned architectural milestone is **v0.6: neural-network operator stack / CNN**, using the Dispatcher introduced in v0.5 rather than bypassing it.