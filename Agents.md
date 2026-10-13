# Agents.md — mini-pytorch v0.14 Development Contract

> Read before modifying branch `v0.14`.

## Branch identity

- Branch: `v0.14`
- Previous: `v0.13`
- Theme: Lowering / Fusion / Inductor-like Concepts
- Status: active until exact final-head CI passes
- Starting baseline: 206 tests

## Required architecture

```text
Graph
  ↓
normalize_graph
  ↓
analyze_graph
  ↓
DCE / safe constant folding
  ↓
fusion planning
  ↓
LoweredProgram
  ↓
LoweredExecutor
```

The v0.11 `GraphInterpreter` remains the semantic reference executor. The v0.13 eager/AOT gradient proofs remain mandatory regression oracles.

## Core rules

- never mutate the source Graph during compiler passes;
- transformations must rebuild deterministic Graph/Value/Node identities;
- explicit analysis owns producer/users/use-count/output-liveness data;
- DCE may remove only operations treated as pure by this educational compiler;
- constant folding is restricted to safe pure constant-only expressions;
- do not fold grad-requiring Tensor constants in v0.14;
- pointwise fusion is limited to `add`, `mul`, `relu`;
- fusion must respect real dependency and use-count structure, not mere adjacency;
- matmul/reduction/conv/pool/AOT helper ops are fusion barriers;
- unsupported but valid graph nodes remain `ExternalOp` where safe;
- fused numerical execution is a reference NumPy mechanism, not generated code;
- CPU/EDU device identity must survive lowering/execution;
- lowering must work on both AOT forward and AOT backward Graphs;
- no performance claims from node-count reduction alone.

## Lowered representation

Expected educational representation:

```text
LoweredProgram
├── ExternalOp
└── PointwiseFusion
      ├── external input Value ids
      ├── ordered fused nodes
      └── output Value ids
```

`PointwiseFusion` may evaluate its small expression DAG with NumPy, but it must not become a second Tensor/backend architecture.

## Required passes

```text
normalize_graph(graph)
analyze_graph(graph)
eliminate_dead_code(graph)
fold_constants(graph)
plan_fusions(graph)
lower_graph(graph)
```

## Testing contract

Required families:

- clone/normalization does not mutate source;
- producer/users/use-count/liveness;
- dead branch removal;
- safe constant folding;
- branch prevents unsafe fusion;
- pointwise chain fusion;
- reduction/matmul barriers;
- deterministic lowering/explain output;
- input-count and malformed-graph failures;
- CPU/EDU execution;
- GraphInterpreter vs LoweredExecutor equality;
- AOT forward lowering;
- AOT backward lowering / gradient equality;
- all historical 206 tests remain green.

Run:

```bash
python -m unittest discover -s tests -v
```

## Documentation contract

Before freeze update:

```text
README.md
version.md
Agents.md
MANIFEST.txt
docs/LOWERING_FUSION.md
examples/17_lowering_fusion.py
tests/test_lowering.py
```

## Explicit non-goals

- production TorchInductor;
- Triton/CUDA/C++ code generation;
- GPU scheduling/autotuning;
- symbolic shapes;
- arbitrary mutation/alias reasoning;
- memory planning/buffer reuse;
- benchmark-backed performance optimization.

## Definition of Done

Freeze only after:

- analysis data model exists;
- normalization/DCE/constant folding are correct and source-preserving;
- explicit deterministic fusion groups exist;
- `LoweredProgram` and `LoweredExecutor` exist;
- forward and backward AOT Graphs lower correctly;
- stats/explain surface exists;
- tests/docs/example/manifest are complete;
- exact final `v0.14` HEAD CI is green;
- validated tree is integrated into `main`.
