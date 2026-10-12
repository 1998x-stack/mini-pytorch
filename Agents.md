# Agents.md — mini-pytorch Development Contract

> **Mandatory instruction for every future iteration:** before changing code, tests, docs, branches, version scope, or architecture, read this file first.
>
> This file is an engineering contract. It is not optional guidance.

---

# 1. Repository mission

mini-pytorch is an educational, NumPy-first framework for understanding the mechanisms behind real PyTorch.

The project optimizes for:

1. architectural clarity;
2. correctness;
3. explicit invariants;
4. readable reference implementations;
5. tests that demonstrate mechanisms;
6. conceptual mapping to real PyTorch.

It does **not** optimize for:

- PyTorch API completeness;
- raw performance before architecture is clear;
- copying upstream PyTorch implementation details;
- adding unrelated features simply because real PyTorch has them.

The guiding principle is:

```text
one version
   ↓
one major mental model
   ↓
readable mechanism
   ↓
explicit tests
   ↓
end-to-end example
```

---

# 2. Mandatory reading order before any work

Every agent must determine the current branch first.

## If current branch is `main`

Read in this order:

1. `Agents.md`
2. `Plan.md`
3. current `version.md` if present
4. `docs/ROADMAP.md`
5. architecture docs relevant to the subsystem being changed
6. existing tests for that subsystem

`Plan.md` is the canonical forward architecture and version sequencing document.

## If current branch is `v0.x` / `v1.x`

Read in this order:

1. `Agents.md`
2. `version.md`
3. relevant source files
4. relevant tests

Do not assume a historical version branch should contain features from a later version.

---

# 3. Branch model

## 3.1 `main` invariant

**`main` must always be the newest integrated, validated version of the project.**

An active version branch is allowed for focused work, but once a commit is validated and represents desired latest behavior, it must be synchronized into `main` immediately.

Forbidden state:

```text
active version branch: newer working code
main: stale older code
```

Required state:

```text
active version branch ── validated increment ──► main
                                             main stays newest
```

## 3.2 Version branches

Version branches exist as learning snapshots.

Examples:

- `v0.1`: Tensor / Autograd / Module / basic optimizer MVP
- `v0.2`: optimizer state + scheduler architecture
- `v0.3`: data pipeline + MNIST classification
- future versions follow `Plan.md`

Every version branch must contain:

- `version.md`
- `Agents.md`

## 3.3 Historical branches

A completed historical branch must not receive future-version functionality.

Allowed changes after freeze:

- documentation correction;
- test correction that preserves version scope;
- narrowly scoped bug fix required for the version’s own promised behavior.

Not allowed:

- backporting future subsystems merely for convenience;
- rewriting the branch so it no longer represents its learning milestone.

---

# 4. Version lifecycle

Every new version follows this process.

## Phase A — plan

Before coding:

1. identify the single primary mental model;
2. explain why this version depends on the previous version;
3. define subsystem ownership and dependency direction;
4. define explicit non-goals;
5. define acceptance criteria;
6. create/update `version.md`;
7. verify the version still matches `Plan.md`.

## Phase B — implement a reference mechanism

Implement the smallest clear mechanism first.

Priority:

```text
correctness
  ↓
readability
  ↓
testability
  ↓
architecture
  ↓
performance
```

If an optimized implementation obscures the idea, keep a readable reference implementation as well.

## Phase C — validate invariants

Tests must prove architectural behavior, not only final numerical output.

Examples:

- Autograd: graph structure and gradient accumulation
- Storage/views: shared storage identity and stride behavior
- Dispatcher: dispatch priority and redispatch
- DataLoader: sampler determines order independently of Dataset
- Distributed: replicas remain consistent after gradient synchronization
- Compiler: eager and compiled paths produce matching results

## Phase D — integrate

After validation:

1. update docs;
2. update `version.md` delta/status;
3. synchronize the validated state into `main`;
4. verify `main` is not behind the active branch.

## Phase E — freeze

When Definition of Done is satisfied:

- version branch becomes a historical learning snapshot;
- normal feature development moves to a new version branch created from latest `main`.

---

# 5. New version branch procedure

When starting `vX.Y`:

1. verify `main` is the newest validated state;
2. create `vX.Y` from current `main`;
3. create/update `version.md` immediately;
4. ensure `Agents.md` exists on the new branch;
5. state the delta from the previous version;
6. implement only that version’s scope;
7. synchronize every validated increment back to `main`;
8. freeze the branch after its Definition of Done.

Never create a future version branch from an old version branch when `main` contains newer integrated work.

Correct:

```text
latest main
   └──► v0.4
```

Incorrect:

```text
v0.2
  └──► v0.4
```

when `main` already contains v0.3 work.

---

# 6. Required `version.md` contract

Every version branch must have `version.md` at repository root.

It must include:

1. branch/version identity;
2. base or historical commit information;
3. previous version;
4. version theme;
5. core difference from previous version;
6. mechanisms added;
7. mechanisms intentionally unchanged;
8. architecture diagram or mental model;
9. important source files;
10. tests and examples;
11. learning goals;
12. non-goals;
13. remaining work if active;
14. next-version direction.

The most important section is always:

> **What changed relative to the previous version, and why?**

---

# 7. Architecture rules

## 7.1 Preserve dependency direction

Preferred high-level dependencies:

```text
Tensor / Storage
      │
      ▼
Operator / Dispatcher
      │
 ┌────┴─────┐
 ▼          ▼
Backend   Autograd
      │
      ▼
      nn
      │
      ▼
    optim

Dataset / Sampler / DataLoader
      │
      └────────────► training code
```

Avoid circular imports between core subsystems.

## 7.2 Separate policy from mechanism

Examples:

- Dataset stores/accesses samples; Sampler chooses indices.
- Optimizer updates parameters; Scheduler changes optimizer learning-rate policy.
- Tensor stores semantic metadata; Dispatcher selects operator implementation.
- Backend kernel computes; Autograd records differentiation behavior.

Do not merge independent responsibilities merely to reduce file count.

## 7.3 Prefer conceptual PyTorch compatibility

When referencing real PyTorch, reproduce the conceptual boundary rather than source-level complexity.

Important upstream concepts include:

- TensorImpl + Storage + sizes/strides/storage offset;
- operator registry + Dispatcher + DispatchKey;
- dynamic autograd Nodes/Functions;
- Module parameters and buffers;
- Optimizer parameter groups/state;
- Dataset/Sampler/DataLoader separation;
- DistributedSampler + DDP gradient synchronization;
- Dynamo graph capture + guards + graph breaks;
- AOTAutograd/AOTDispatcher;
- Inductor-like graph lowering/fusion.

Do not copy PyTorch source code. Implement original educational equivalents.

---

# 8. Scope discipline

Each version should introduce one primary architectural idea.

Before adding a feature, ask:

1. Does this feature explain the current version’s mental model?
2. Is it required by an acceptance criterion?
3. Is it a dependency for a current mechanism?
4. Would adding it now hide a more fundamental subsystem that should exist first?

If the answer indicates future scope, record it in `Plan.md` or `version.md` rather than implementing it immediately.

Example:

Do not add multiprocessing DataLoader merely because real PyTorch supports `num_workers` before single-process iterator/fetcher lifecycle is explicit.

Do not add CUDA-specific branches throughout Tensor before Dispatcher/Device abstractions exist.

---

# 9. API design rules

Use PyTorch-like APIs when they help transfer learning, but do not mimic every edge case.

Good reasons to match PyTorch-like API shape:

- makes conceptual mapping obvious;
- reduces cognitive load;
- demonstrates why an abstraction exists.

Good reasons to simplify:

- production behavior would require a large unrelated subsystem;
- edge-case compatibility obscures the core algorithm;
- the educational invariant can be expressed more clearly with a smaller API.

Every material simplification should be documented.

---

# 10. Testing requirements

Every new subsystem must include, where applicable:

## A. Happy path

Expected normal behavior.

## B. Invariant/property tests

Tests for the mechanism itself.

## C. Error-path tests

Invalid combinations must fail clearly.

## D. Numerical/reference validation

Examples:

- finite-difference gradcheck;
- compare Conv2d to a trusted reference implementation;
- compare eager vs compiled result;
- optimizer convergence/state round trip.

## E. Integration test

Connect the subsystem to the surrounding framework.

Preferred command for current unittest suite:

```bash
python -m unittest discover -s tests -v
```

Do not claim the full test suite passed unless it was actually run successfully on the exact code being reported.

---

# 11. Documentation requirements for code changes

Update documentation when a change modifies:

- public API;
- architecture boundary;
- version scope;
- branch/version policy;
- learning sequence;
- important invariants;
- example usage.

For a version change, review at minimum:

- `version.md`
- `Agents.md`
- `Plan.md` on main if sequencing changed
- relevant file under `docs/`
- README when user-facing setup/usage changes

---

# 12. Commit rules

Prefer commits that express architectural intent.

Examples:

```text
feat(data): add sampler abstraction
feat(storage): introduce explicit storage metadata
feat(dispatch): add operator registry and CPU dispatch key
test(storage): cover view aliasing and contiguous copies
docs: explain v0.4 tensor memory model
```

Avoid mixing unrelated subsystems in one commit.

A sequence of small coherent commits is preferable to a giant opaque rewrite.

---

# 13. Backward compatibility rules

Before changing an existing public API:

1. identify who uses it in examples/tests;
2. decide whether compatibility is pedagogically valuable;
3. migrate all internal callers atomically;
4. document intentional breakage.

Historical version branches should preserve their own API snapshot even if later versions evolve.

---

# 14. Data, generated output and dependencies

Do not commit generated datasets or training runs.

Typical generated paths such as MNIST raw data, run logs, plots and checkpoints should remain ignored unless a tiny fixture is specifically required for tests.

Add dependencies only when they are directly justified by current version goals.

NumPy remains the default numerical reference backend unless a later version explicitly introduces another backend.

---

# 15. Performance rule

Never sacrifice the educational reference path solely for speed.

Preferred progression:

```text
readable reference
       ↓
correctness tests
       ↓
profile
       ↓
optimized educational path
       ↓
compare both implementations
```

Examples:

- Conv2d loops before im2col;
- eager pointwise operators before fusion;
- simple gradient synchronization before buckets.

---

# 16. Real PyTorch reference policy

When architecture depends on current PyTorch behavior, prefer official sources:

- PyTorch repository source/docs;
- docs.pytorch.org;
- official PyTorch design notes/tutorials.

Relevant conceptual anchors:

- `c10/core/TensorImpl.h`
- PyTorch Dispatcher documentation/walkthrough
- autograd documentation/design notes
- DDP documentation
- torch.compiler programming model

Use upstream sources to verify concepts. Do not reproduce long source passages or copy implementation code.

---

# 17. Definition of Done for any version

A version can be frozen only when all applicable items are true:

- [ ] primary mental model is clearly stated;
- [ ] implementation is complete for promised scope;
- [ ] unit tests pass;
- [ ] invariant tests pass;
- [ ] error-path tests exist;
- [ ] numerical/reference validation exists where relevant;
- [ ] integration example works;
- [ ] `version.md` accurately describes the delta;
- [ ] `Agents.md` exists;
- [ ] architecture docs are updated;
- [ ] package imports cleanly;
- [ ] generated artifacts are not committed;
- [ ] `main` contains the newest validated state;
- [ ] next version’s dependency rationale is documented.

---

# 18. Current branch directive: `main`

`main` is the integration branch and canonical newest state.

For work performed directly on `main`:

1. read `Plan.md` before choosing the next subsystem;
2. do not bypass the current version’s dependency sequence;
3. keep documentation and tests synchronized with code;
4. if a version branch is active, ensure validated work from that branch is represented here;
5. never intentionally reset `main` to an older version snapshot.

Current roadmap priority is defined by `Plan.md`; at present, finish the v0.3 data-pipeline scope before beginning the v0.4 Storage/stride/view subsystem.
