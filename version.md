# mini-pytorch v0.14

## Version identity

- **Branch:** `v0.14`
- **Previous version:** `v0.13`
- **Theme:** Lowering / Fusion / Inductor-like Concepts
- **Status:** active development
- **Package version target:** `0.14.0`
- **Starting regression baseline:** **206/206 tests passed**

v0.14 asks:

> Once forward and backward are explicit Graph programs, how can compiler passes analyze and transform those graphs into a lower-level execution plan without changing semantics?

## Core delta from v0.13

```text
v0.13
forward Graph
+
backward Graph

v0.14
Graph
  ↓
normalize / analyze
  ↓
DCE / constant folding
  ↓
fusion grouping
  ↓
LoweredProgram
  ↓
LoweredExecutor
```

The original `GraphInterpreter` remains the reference executor.

## Planned compiler passes

```text
normalize_graph(graph)
analyze_graph(graph)
eliminate_dead_code(graph)
fold_constants(graph)
plan_fusions(graph)
lower_graph(graph)
```

## Initial fusion scope

Only pure pointwise chains are fusion candidates:

```text
aten::add.Tensor
aten::mul.Tensor
aten::relu.default
```

Initial barriers remain external operations:

```text
matmul
sum
conv / pool
aot::unbroadcast
aot::sum_backward
aot::transpose2d
aot::relu_backward
```

## Required correctness proof

For supported programs:

```text
GraphInterpreter(graph)
==
LoweredExecutor(lower_graph(graph))
```

and for AOT programs:

```text
AOT forward Graph → lowered forward output
AOT backward Graph → lowered input gradients
```

must match the v0.13 reference behavior.

## Design rules

- source Graphs are never mutated by passes;
- analysis and fusion decisions are inspectable;
- DCE assumes only the documented pure logical-op subset;
- constant folding is restricted to safe pure constants;
- pointwise fusion must respect use-count/dependency structure;
- unsupported nodes remain `ExternalOp` when safe;
- CPU/EDU logical device identity must be preserved;
- reduced node count is not a performance claim;
- no Triton/CUDA code generation in this version.

## Non-goals

- production TorchInductor;
- generated C++/CUDA/Triton kernels;
- GPU scheduling/autotuning;
- symbolic-shape lowering;
- arbitrary mutation/alias optimization;
- runtime memory planning or performance claims.

## Next version

v0.15 will study runtime/performance mechanics after a stable lowered execution representation exists.
