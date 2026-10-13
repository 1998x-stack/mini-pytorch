# Agents.md — mini-pytorch v1.0 Release Candidate Contract

> Read before modifying branch `v1.0`.

## 1. Branch identity

- Branch: `v1.0`
- Previous: `v0.15`
- Theme: Integrated Architecture
- Status: **release candidate**
- Package version: `1.0.0`
- Starting baseline: 247 tests / OK
- Current validated baseline: **251 tests / OK**
- Validated framework/CI commit: `d49b1872274f790303d719034074b75c270a18c2`

## 2. Current directive

The v1.0 mechanism/documentation scope is complete.

Do not add another subsystem or optional feature before freeze.

Allowed changes before freeze are limited to:

- release-document corrections;
- fixes for failures found by exact-head CI;
- corrections required for the integrated example to remain executable;
- consistency fixes inside already-audited v1.0 boundaries.

## 3. Frozen architecture target

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

Preserve the ownership boundaries documented in `docs/ARCHITECTURE_V1.md`.

## 4. Release-candidate evidence

Exact framework commit `d49b1872274f790303d719034074b75c270a18c2` passed:

```text
python -m unittest discover -s tests -v
→ Ran 251 tests
→ OK
```

and:

```text
python examples/19_integrated_architecture.py
→ checkpoint restore PASS
→ graph cache hit/reuse PASS
→ planned output matches eager PASS
→ planned gradients match eager PASS
```

GitHub Actions environment:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

## 5. Release artifacts that must remain present

```text
README.md
version.md
Agents.md
MANIFEST.txt
pyproject.toml
mini_pytorch/_version.py

docs/ARCHITECTURE.md
docs/ARCHITECTURE_V1.md
docs/API_AUDIT.md
docs/LEARNING_PATH.md
docs/PYTORCH_MAPPING.md
docs/ROADMAP.md

examples/19_integrated_architecture.py
tests/test_v1_integration.py
```

## 6. API policy

The v1.0 API audit intentionally chose **no breaking surface rewrite**.

Do not remove/rename historical public learning concepts merely for cosmetic similarity with PyTorch.

Keep these distinctions explicit:

```text
project educational analogue
!=
upstream source/API parity
```

## 7. Integrated example policy

The integrated example separates:

```text
eager training
runtime/checkpoint state
compiler specialization
AOT/lowering/planned runtime
```

Do not force DDP/checkpoint/compiler execution into one unrealistic production loop. The architectural integration is expressed by ownership boundaries and executable cross-layer proofs.

## 8. CI policy

Workflow must continue to run for:

```text
main
v*
```

Required release gate:

```bash
python -m unittest discover -s tests -v
python examples/19_integrated_architecture.py
```

Never claim a later HEAD is validated by the earlier `d49b1872...` run. Every release-candidate/frozen/main HEAD gets its own CI result.

## 9. Explicit non-goals

Do not add before freeze:

- CUDA/Triton/codegen;
- new backend families;
- symbolic shapes;
- worker/prefetch DataLoader runtime;
- NCCL/network distributed runtime;
- production allocator/profiler;
- whole-Python compiler;
- PyTorch feature parity.

## 10. Freeze workflow

```text
current release-candidate docs commit
      ↓
exact HEAD CI green
      ↓
change only version.md + Agents.md to frozen status
      ↓
exact frozen HEAD CI green
      ↓
verify main...v1.0 is pure fast-forward
      ↓
fast-forward v1.0 into main
      ↓
restore canonical main README / Agents / Plan
      ↓
exact main HEAD CI green
```

## 11. Frozen branch rule after release

After v1.0 freezes, permitted branch changes are only:

- documentation corrections;
- tests clarifying already-promised behavior;
- narrow fixes inside the frozen v1.0 contract.

Future architecture work must start from latest `main` and first justify a new missing mental model.
