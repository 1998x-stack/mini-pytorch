# Agents.md — mini-pytorch Canonical Development Contract

> Mandatory for future work on `main` or any new version branch.
>
> `main` contains the newest integrated, validated framework state. Historical version branches are frozen learning snapshots.

## 1. Current project state

- Latest integrated release: **v1.0.0**
- Frozen `v1.0` release baseline: **251 tests / OK**
- Post-v1.0 reviewed `main` implementation baseline: **292 tests / OK**
- Reviewed implementation commit: `28a9071dbd89d2af553a328c5288455facb36209`
- Integrated example: `examples/19_integrated_architecture.py` / PASS
- Original v0.x → v1.0 learning sequence: **complete**

The 292-test baseline includes maintenance/hardening added after the frozen v1.0 learning snapshot. It does **not** redefine the historical v1.0 branch as a 292-test release.

There is intentionally no automatic `v1.1` roadmap.

## 2. Mission

mini-pytorch is a NumPy-first educational framework for exposing deep-learning framework architecture one ownership boundary at a time.

Priority order:

```text
architectural clarity
      ↓
correctness / invariants
      ↓
inspectability
      ↓
reference proofs / tests
      ↓
measurement
      ↓
optimization
```

Do not optimize for API breadth or upstream-source mimicry.

## 3. Integrated v1.0 architecture

```text
Tensor / Storage / Device
      ↓
Dispatcher / Backend
      ↓
AutogradEngine
      ↓
Module / Optimizer / Data
      ↓
Runtime / Checkpoint / RNG
      ↓
Distributed / DDP
      ↓
Capture / Guards / Cache
      ↓
AOT forward/backward Graphs
      ↓
Analysis / Lowering / Fusion
      ↓
LoweredProgram
      ↓
Lifetime / MemoryPlan / ExecutionPlan
      ↓
PlannedExecutor
```

The detailed ownership/failure audit lives in `docs/ARCHITECTURE_V1.md`.

## 4. Preserved architecture invariants

### Tensor / memory

```text
Storage owns allocation + shared version
Tensor owns shape / stride / offset + autograd-facing state
Device describes logical placement
```

Compiler/runtime planning must not redefine Tensor view/storage semantics.

### Dispatcher / backend

```text
logical operator
      ↓
Dispatcher policy
      ↓
backend kernel
```

Backend numerical kernels do not own Module, Optimizer, DDP, GraphCache or checkpoint policy.

### Autograd

`AutogradEngine` owns eager backward execution. Saved-tensor version safety, graph lifetime, hooks and leaf-grad ownership remain explicit separate concerns.

### Training/runtime

Module organizes parameters/buffers. Optimizer consumes gradients. Data provides batches. Runtime owns reproducible checkpoint/RNG/progress state.

### Distributed

DDP/Reducer/ProcessGroup synchronize gradients outside optimizer algorithms and numerical kernels.

### Compiler

```text
capture
  ↓
Graph IR
  ↓
guards/cache
  ↓
AOT differentiation
  ↓
analysis/transforms/lowering
  ↓
LoweredProgram
```

Graph IR does not bake in backend/Autograd dispatch policy. Source Graphs are not silently mutated by transformation passes.

### Planned runtime

```text
LoweredProgram = what executes
ExecutionPlan = how runtime resources are assigned
```

ExternalOps stay Dispatcher-owned; safe temporary reuse remains constrained by liveness/ownership rules.

## 5. Historical branch rule

Branches `v0.1` through `v1.0` are learning snapshots.

Allowed changes on frozen branches:

- documentation corrections;
- tests clarifying already-promised behavior;
- narrow fixes inside the historical version contract.

Future mechanisms do not belong on a frozen branch.

## 6. Main branch rule

`main` must remain the newest integrated validated framework state.

Do not intentionally leave validated framework work only on a version branch after release integration.

Canonical project docs on `main` are:

```text
README.md
Agents.md
Plan.md
REVIEW.md
docs/ARCHITECTURE_V1.md
docs/LEARNING_PATH.md
docs/ROADMAP.md
docs/PYTORCH_MAPPING.md
docs/REVIEW_FINDINGS.md
docs/REVIEW_MATRIX.md
```

`REVIEW.md` and the two `docs/REVIEW_*` files record post-v1.0 maintenance findings and exact-head validation evidence. They do not change the frozen historical scope of `v1.0`.

## 7. Post-v1.0 change gate

Before creating another version, write down all of the following:

1. **Missing mental model** — what important architecture question cannot v1.0 answer?
2. **Owning boundary** — which existing layer should own the new mechanism?
3. **Minimal proof** — what is the smallest executable implementation that teaches it?
4. **Invariants** — what must remain true across existing layers?
5. **Non-goals** — what tempting adjacent work is explicitly excluded?
6. **Reference validation** — what existing layer remains the correctness oracle?

If those cannot be stated clearly, do not create a new version.

## 8. New-version workflow, if justified

```text
validated main
   ↓
define new mental model in Plan.md
   ↓
create vX.Y from exact main HEAD
   ↓
create branch version.md + Agents.md immediately
   ↓
implement readable reference mechanism
   ↓
add invariant/reference/integration tests
   ↓
complete docs/example
   ↓
exact release-candidate HEAD CI green
   ↓
freeze branch
   ↓
exact frozen HEAD CI green
   ↓
fast-forward into main
   ↓
restore canonical main README / Agents / Plan
   ↓
exact main HEAD CI green
```

Never create a new feature branch from an older frozen version when `main` is newer.

## 9. Testing contract

Full regression:

```bash
python -m unittest discover -s tests -v
```

The current CI additionally runs:

```bash
python examples/19_integrated_architecture.py
```

For test-count claims, distinguish the snapshot being described:

```text
frozen v1.0 branch: 251 tests / OK
reviewed post-v1.0 main implementation: 292 tests / OK
```

Every meaningful change should include, where applicable:

- happy path;
- invariant/property proof;
- error-path proof;
- numerical/reference validation;
- cross-layer integration proof.

Never claim a test count or performance result unless the exact reported code state actually ran.

## 10. Performance-reporting contract

Performance evidence requires:

```text
explicit warmup
repeated samples
same workload/input structure
median/range or distribution
planning/compile time separated from steady-state execution
allocation metrics separated from wall-clock timing
```

A lower instruction/allocation count is not automatically a speedup.

## 11. PyTorch mapping contract

`docs/PYTORCH_MAPPING.md` maps responsibilities and concept families.

Never turn that mapping into unsupported claims of:

- source equivalence;
- exact API parity;
- edge-case parity;
- production performance parity.

Upstream PyTorch can evolve while frozen historical branches remain historical snapshots.

## 12. Explicit original-project non-goals

The integrated v1.0 learning release intentionally does not include:

- real CUDA/Triton codegen;
- symbolic dynamic-shape compiler;
- multiprocessing/prefetch/pinned-memory DataLoader;
- NCCL/network distributed execution;
- production caching allocator;
- whole-Python bytecode compiler;
- complete higher-order autodiff;
- feature parity with PyTorch.

Future work may revisit one of these only after it passes the post-v1.0 change gate.

## 13. Current directive

The project is currently in **stable integrated v1.0 architecture + post-v1.0 maintenance state**.

Default action is maintenance, study, benchmarking or documentation refinement—not automatic feature expansion.
