# Agents.md — mini-pytorch v0.14 Frozen Contract

> Historical learning snapshot. Read before changing branch `v0.14`.

## Branch identity

- Branch: `v0.14`
- Previous: `v0.13`
- Theme: Lowering / Fusion / Inductor-like Concepts
- Status: **frozen historical learning snapshot**
- Package: `0.14.0`
- Validated baseline: **221 tests / OK**

## Frozen architecture

```text
Graph
  ↓
normalize_graph
  ↓
analyze_graph
  ↓
DCE / safe constant folding
  ↓
use-aware fusion planning
  ↓
LoweredProgram
  ↓
LoweredExecutor
```

`GraphInterpreter` remains the semantic reference. v0.13 explicit forward/backward Graphs remain the gradient correctness reference.

## Frozen invariants

- compiler passes do not mutate source Graphs;
- transformed Graphs use deterministic rebuilt Value/Node ids;
- analysis explicitly records producer/users/use-count/output/liveness;
- DCE assumes only the documented pure logical-op model;
- constant folding is restricted to pure constant-only add/mul/relu;
- grad-requiring constants/outputs are not folded;
- pointwise fusion is limited to add/mul/relu;
- fusion respects dependency and occurrence-level use-count structure;
- matmul/reduction/conv/pool/AOT helper operators remain barriers;
- `LoweredProgram` consists of `ExternalOp` and `PointwiseFusion` instructions;
- ExternalOps execute through Dispatcher;
- PointwiseFusion uses readable NumPy reference execution and preserves CPU/EDU logical device identity;
- `LoweredExecutor` executes under `no_grad` and is first-order;
- source-to-lowered Value mapping preserves the v0.13 saved-value ABI;
- the same lowering pipeline accepts AOT forward and backward Graphs;
- reduced node/instruction count is not a performance claim.

## Validated behavior

The 221-test baseline covers:

- analysis/liveness/use counts;
- deterministic normalization;
- DCE;
- constant folding and its grad-safety boundary;
- fusion chains and branch barriers;
- fusion barriers;
- CPU/EDU lowered execution;
- external Dispatcher execution;
- deterministic stats/explain output;
- AOT forward saved-value remapping;
- lowered backward-Graph gradient equality;
- malformed graph / input-count / mixed-device errors;
- every historical regression from v0.1-v0.13.

## Historical branch rule

Do not add future runtime/performance features to this branch.

Allowed changes after freeze are limited to:

- documentation corrections;
- tests clarifying behavior already promised by v0.14;
- narrow bug fixes that remain strictly inside the v0.14 contract.

The following belong to later branches:

- profiling and benchmark framework;
- Value lifetime intervals;
- memory/buffer planning;
- buffer reuse / preallocation;
- generated kernels;
- symbolic shapes;
- runtime scheduling / autotuning.

## Important files

```text
version.md
README.md
docs/LOWERING_FUSION.md
mini_pytorch/compiler/analysis.py
mini_pytorch/compiler/lowering.py
tests/test_lowering.py
examples/17_lowering_fusion.py
```

## Next architectural dependency

The next version is **v0.15 — Runtime / Performance Mechanics**.

Central question:

> Once a correct LoweredProgram exists, how should the runtime measure its execution, reason about Value lifetimes and allocations, and reuse memory safely based on evidence rather than assumption?
