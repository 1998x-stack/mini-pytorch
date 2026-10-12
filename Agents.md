# Agents.md — mini-pytorch Development Contract

> **Mandatory for every future iteration:** read this file before changing code, tests, documentation, branches, version scope, or architecture.
>
> This is the canonical engineering contract for `main`. Version branches carry branch-specific `Agents.md` files.

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

Do not optimize for API completeness or copy upstream PyTorch source.

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

Never assume a historical branch should contain later-version features.

## 3. Branch invariant

`main` must always be the newest integrated, validated state.

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
integrate validated tree into main
   ↓
restore canonical main docs
```

Forbidden:

```text
version branch has newer validated framework code
main remains stale
```

## 4. Version sequence

```text
v0.1  Tensor / Dynamic Autograd
v0.2  Optimizer state / LR Scheduler
v0.3  Dataset / Sampler / DataLoader / MNIST
v0.4  Storage / stride / views / inplace version safety
v0.5  Operator Registry / Dispatcher / Redispatch
v0.6  Neural-network operator stack / CNN
v0.7  Device / Backend abstraction                     ← next
```

One version should introduce one dominant mental model.

## 5. Current architecture

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
       functional NN ops
             ▼
            nn
             ▼
           optim

Dataset → Sampler → BatchSampler → DataLoader
```

v0.6 proves this stack can support Conv2d, MaxPool2d, Dropout, BatchNorm and CNN training.

## 6. Architecture rules

- separate policy from mechanism;
- avoid circular core imports;
- backend kernels must not depend on Module/Optimizer/DataLoader;
- Autograd policy must not be duplicated inside every backend kernel;
- preserve Storage/stride/view/version invariants;
- high-level `nn.Module` should own state/configuration, not backend loops;
- functional operators should be the bridge from modules to logical operators;
- do not scatter device-specific `if` statements through Tensor in v0.7;
- Device/backend support must extend Dispatcher rather than bypass it.

## 7. Conceptual PyTorch compatibility

Prefer PyTorch-like boundaries when they improve transfer of understanding:

- TensorImpl-like Storage + sizes + strides + offset;
- dynamic Autograd Functions/Nodes;
- Module Parameters and Buffers;
- Optimizer parameter groups/state;
- Dataset/Sampler/DataLoader separation;
- operator schema + Dispatcher + DispatchKey;
- redispatch from wrappers to backend kernels;
- Module → functional operator → Dispatcher for neural-network ops;
- next: Device metadata and backend-owned allocation.

Implement original educational equivalents.

## 8. Testing contract

Every new subsystem should include, where applicable:

- happy path;
- invariant/property tests;
- error-path tests;
- numerical/reference validation;
- integration test.

Full regression:

```bash
python -m unittest discover -s tests -v
```

Never claim a version is validated unless the exact reported code has passed the relevant suite. Before freeze, GitHub Actions must pass on the exact final version-branch head.

Current integrated baseline:

```text
v0.6
90 tests
OK
```

## 9. Documentation contract

When architecture/public API changes, review:

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

A frozen branch may receive only:

- documentation corrections;
- tests clarifying its promised behavior;
- narrowly scoped bug fixes within that version boundary.

Do not backport future subsystems.

## 11. Performance rule

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

Examples already established:

- Conv2d nested loops before im2col;
- eager kernels before future fusion;
- simple backend abstraction before accelerator optimization.

## 12. Generated artifacts and dependencies

Do not commit generated datasets, logs, plots or checkpoints unless a tiny deterministic fixture is explicitly required by tests.

NumPy remains the reference CPU backend. A future optional accelerator backend must not make the project depend on unavailable hardware.

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

`main` contains the completed v0.6 CNN operator milestone.

The next version must be created from current `main` and should be **v0.7: Device / Backend abstraction**.

The central v0.7 question is:

> How can Tensor allocation and operator execution carry explicit device/backend identity without scattering `if device == ...` across the framework?

v0.7 should build on the existing Dispatcher instead of replacing it.