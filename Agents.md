# Agents.md — mini-pytorch Development Contract

> **Mandatory for every future iteration:** read this file before changing code, tests, documentation, branches, version scope, or architecture.
>
> This is the canonical engineering contract for `main`. Version branches carry their own branch-specific `Agents.md`.

---

# 1. Repository mission

mini-pytorch is an educational, NumPy-first framework for understanding the mechanisms behind real PyTorch.

The project optimizes for:

1. architectural clarity;
2. correctness and explicit invariants;
3. readable reference implementations;
4. tests that demonstrate mechanisms rather than only outputs;
5. conceptual mapping to real PyTorch;
6. one major mental model per version.

It does **not** optimize for:

- API completeness;
- raw performance before the mechanism is clear;
- copying upstream PyTorch source code;
- prematurely implementing CUDA/compiler/distributed details;
- adding unrelated features merely because PyTorch has them.

Core principle:

```text
one version
   ↓
one major mental model
   ↓
readable mechanism
   ↓
explicit invariants + tests
   ↓
end-to-end example
   ↓
freeze learning snapshot
```

---

# 2. Mandatory reading order

## On `main`

Read in this order before work:

1. `Agents.md`
2. `Plan.md`
3. `version.md` — latest integrated version
4. architecture docs relevant to the subsystem
5. tests relevant to the subsystem
6. existing implementation being changed

`Plan.md` is the canonical source for **what version comes next and why**.

## On a version branch (`v0.x`, `v1.x`)

Read:

1. branch `Agents.md`
2. branch `version.md`
3. relevant architecture docs
4. relevant tests
5. relevant source

Never assume a historical version should contain a later-version feature.

---

# 3. Branch model

## 3.1 `main` invariant

**`main` must always represent the newest integrated project state.**

Required state:

```text
active version branch
       │
       │ validated increment
       ▼
     main
       │
       └── newest integrated implementation
```

Forbidden long-lived state:

```text
version branch = newer working implementation
main           = stale older implementation
```

Small transient differences while a validation run is executing are acceptable, but completed desired behavior must be integrated immediately.

## 3.2 Version branches

Version branches are educational snapshots, not permanent development trunks.

Every version branch must contain:

- `version.md`
- `Agents.md`
- tests for the introduced mechanism
- architecture documentation when a new subsystem is introduced
- at least one educational example when appropriate

When a version is complete:

1. exact branch CI must pass;
2. `version.md` becomes frozen/completed;
3. branch `Agents.md` becomes frozen;
4. validated implementation is synchronized to `main`;
5. `main/Plan.md` is advanced to the next version;
6. next version branch is created from the newest `main`.

## 3.3 Historical branch policy

Frozen branches may receive only:

- documentation corrections;
- test corrections preserving version semantics;
- narrowly scoped fixes for behavior already promised by that version.

Do not backport later architecture merely to make an older branch “feature complete”.

---

# 4. Current repository state

Completed learning snapshots:

```text
v0.1  Tensor + eager dynamic Autograd + Module + basic Optimizer
v0.2  optimizer state + optimizer families + LR schedulers
v0.3  Dataset + Sampler + BatchSampler + DataLoader + MNIST classification
v0.4  Storage + stride + views + contiguous + version counter
```

Current integrated package version:

```text
0.4.0
```

**Next architectural milestone:** `v0.5` — operator schema / registry / Dispatcher-like routing.

Do not begin v0.6 CNN work until v0.5 has a validated operator-routing boundary.

---

# 5. How to choose scope

Before implementation, state the single learning question the version answers.

For v0.5 the question is:

> How can one logical Tensor operator be registered once and routed to the correct implementation without hard-coding backend/autograd decisions in every Tensor method?

A proposed change belongs in the current version only if it directly helps answer that question or is necessary to preserve existing behavior.

If a feature belongs to a later version, document it in `Plan.md`; do not implement it early.

---

# 6. Reference-real-PyTorch rule

Use real PyTorch as an architectural reference, not as source to copy.

Preferred mapping:

```text
mini-pytorch concept
        ↓
public PyTorch behavior
        ↓
real PyTorch subsystem boundary
```

Examples:

```text
Storage/stride/view        → Tensor/TensorImpl-like metadata concepts
operator registry          → operator schema/registration concept
Dispatcher                 → dispatch-key based operator routing
Autograd Function/GradFn   → autograd graph/node concept
Module/Parameter           → nn.Module registration model
ProcessGroup/DDP           → collective gradient synchronization
Graph IR                   → eager-to-graph compiler boundary
```

Do not reproduce substantial upstream source text. Re-derive a small educational implementation from the conceptual model.

---

# 7. Dependency-direction rule

Keep subsystem dependencies explicit.

Target direction:

```text
Storage / Tensor metadata
          ↓
Operator schema / Dispatcher
          ↓
Backend kernels + Autograd wrappers
          ↓
NN layers / losses
          ↓
Optimizer / scheduler

Dataset / Sampler / DataLoader
          ↓
Training runtime

Distributed / compiler layers
          ↓
consume the stable lower layers
```

Avoid circular dependencies and hidden global behavior unless the version specifically exists to teach a registry/global-runtime concept.

---

# 8. Implementation rule

For a new subsystem:

1. define the public mental model;
2. define invariants;
3. implement the smallest reference mechanism;
4. add error paths;
5. integrate one existing operator/path;
6. test compatibility with earlier versions;
7. add an educational example;
8. document mapping to real PyTorch.

Prefer:

```text
correct + readable
      ↓
well tested
      ↓
architecturally explicit
      ↓
profiled
      ↓
optimized only when optimization teaches something
```

---

# 9. Testing contract

The default regression command is:

```bash
python -m unittest discover -s tests -v
```

Every new mechanism needs, where applicable:

- happy-path tests;
- invariant tests;
- shape/state tests;
- error-path tests;
- backward/gradient tests;
- interaction tests with previous subsystems;
- one end-to-end integration test or example.

Never claim the whole suite passes unless the exact branch/commit was executed.

GitHub Actions is the canonical remote validation environment for release/freeze decisions.

When CI finds a regression, fix the mechanism rather than weakening tests unless the test itself encoded an incorrect invariant.

---

# 10. Mutation and compatibility rule

v0.4 established Storage aliasing and version counters. Later versions must preserve these invariants:

- view-producing operators must retain correct alias semantics;
- copying kernels must produce independent Storage when appropriate;
- tracked inplace writes must bump the shared version counter;
- Autograd wrappers must not bypass saved-version correctness;
- scalar `shape == ()` must stay scalar;
- optimizer/state-loading writes must remain version-aware.

Any Dispatcher/backend work must preserve these semantics rather than reimplementing simplified copies that bypass them.

---

# 11. Documentation contract

`version.md` answers:

- what this version is;
- how it differs from the previous version;
- what mechanisms it adds;
- what stays unchanged;
- what is explicitly out of scope;
- how it was validated;
- what the next version is.

`Plan.md` answers:

- why versions occur in this order;
- what future subsystem boundaries are planned;
- each version’s learning objective and Definition of Done.

Architecture docs (`docs/*.md`) answer:

- how a subsystem works internally;
- invariants and examples;
- mapping to real PyTorch concepts.

Whenever a version is frozen, all three layers must agree.

---

# 12. Commit / integration discipline

Prefer focused commits such as:

```text
feat(dispatch): add operator schema registry
feat(dispatch): introduce dispatch keys
feat(dispatch): route add through dispatcher
test(dispatch): cover priority and fallback
docs(v0.5): explain dispatcher architecture
```

Avoid mixing unrelated features in one commit.

Before freezing:

```text
implementation
  ✓
tests
  ✓
example
  ✓
docs
  ✓
version.md
  ✓
remote CI
  ✓
main synchronization
  ✓
```

---

# 13. Current directive

The next normal feature work should create `v0.5` **from the latest `main`**.

v0.5 should focus on:

```text
OperatorSchema
OperatorRegistry
DispatchKey
DispatchKeySet
Dispatcher
kernel registration
dispatch priority
redispatch/fallback
selected Tensor operator migration
Autograd/backend routing demonstration
```

Do **not** add Conv2d, CUDA, DDP or compiler work to v0.5 merely to make the demo larger.

The intended outcome is that, after v0.5, a reader can explain why a framework needs a Dispatcher before it can cleanly support multiple backends and execution modes.