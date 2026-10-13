# Agents.md — mini-pytorch v1.0 Frozen Release Contract

> Branch `v1.0` is a frozen historical/integrated learning snapshot.

## 1. Branch identity

- Branch: `v1.0`
- Previous: `v0.15`
- Theme: Integrated Architecture
- Status: **frozen**
- Package version: `1.0.0`
- Frozen test baseline: **251 tests / OK**
- Validated release-candidate HEAD: `7a8391f0959e0671e2bf40720e67296793048947`

## 2. Frozen architecture

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

The ownership/failure boundaries in `docs/ARCHITECTURE_V1.md` are part of the frozen v1.0 contract.

## 3. Validation contract

The release candidate passed:

```bash
python -m unittest discover -s tests -v
python examples/19_integrated_architecture.py
```

with:

```text
251 tests / OK
integrated architecture example / PASS
```

The exact frozen HEAD must pass the same workflow before `main` integration.

## 4. Frozen artifacts

Key release artifacts:

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

## 5. Rules for future changes on this branch

Allowed only:

- documentation corrections;
- tests that clarify already-promised v1.0 behavior;
- narrowly scoped fixes inside the frozen architecture contract.

Forbidden:

- new subsystems;
- new backend families;
- compiler roadmap expansion;
- feature-parity additions;
- future-version mechanisms.

## 6. Compatibility / mapping rule

Do not rewrite v1.0 history because upstream PyTorch later changes terminology or implementation.

`docs/PYTORCH_MAPPING.md` records conceptual correspondence, not a source/API parity promise.

## 7. Future development rule

Future development must:

```text
start from latest validated main
      ↓
identify a genuinely missing mental model
      ↓
write a new roadmap/branch contract
      ↓
create a new version branch
```

There is no automatic `v1.1` queue.

## 8. Integration sequence

After exact frozen-head CI succeeds:

```text
verify main → v1.0 is pure fast-forward
      ↓
fast-forward main to frozen v1.0
      ↓
restore canonical main README / Agents / Plan
      ↓
run exact main-head CI
```

The frozen `v1.0` branch itself must then remain a stable learning snapshot.
