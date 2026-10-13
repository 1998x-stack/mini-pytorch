# mini-pytorch v0.14

## Version identity

- **Branch:** `v0.14`
- **Previous version:** `v0.13`
- **Theme:** Lowering / Fusion / Inductor-like Concepts
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.14.0`
- **Validated regression baseline:** **221/221 tests passed**

v0.14 answers:

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
DCE / safe constant folding
  ↓
use-aware pointwise fusion
  ↓
LoweredProgram
  ↓
LoweredExecutor
```

`GraphInterpreter` remains the semantic reference executor.

## Implemented analysis layer

```text
ValueAnalysis
GraphAnalysis
analyze_graph(graph)
iter_value_ids(spec)
```

For every graph Value the analysis records:

```text
producer_node_id
user_node_ids
use_count
is_graph_output
is_live
```

`use_count` counts occurrences, not only distinct user nodes. Therefore a value used twice by one node is correctly treated as having two uses for fusion safety.

## Implemented transformation passes

```text
normalize_graph(graph)
eliminate_dead_code(graph)
fold_constants(graph)
plan_fusions(graph)
lower_graph(graph)
```

### Normalization

- deterministic deep Graph rebuild;
- no mutation of the source Graph;
- graph input/output ordering preserved;
- constants, metadata, operators and kwargs preserved;
- validation before and after transformation.

### Dead Code Elimination

DCE computes reverse reachability from graph outputs and removes unreachable pure nodes and constants.

The v0.14 purity assumption is deliberately limited to the current educational logical-op Graph model; it is not general Python side-effect analysis.

### Safe constant folding

Initial foldable set:

```text
aten::add.Tensor
aten::mul.Tensor
aten::relu.default
```

Folding requires constant-only Tensor dependencies on one logical device and rejects grad-requiring constants/outputs. Fold evaluation reuses Dispatcher under `no_grad` and snapshots the result as a new `TensorConstant`.

## Fusion model

Initial fusible set:

```text
aten::add.Tensor
aten::mul.Tensor
aten::relu.default
```

Fusion is intentionally conservative. A linear chain extends only when the current intermediate:

```text
use_count == 1
AND
is not a graph output
AND
is consumed by the next fusible node
```

Branching/shared values therefore stop fusion.

Explicit compiler data:

```text
FusionGroup
├── id
├── node_ids
├── input_value_ids
└── output_value_ids
```

## Fusion barriers

The following remain external in v0.14:

```text
matmul
sum
conv2d
max_pool2d
aot::unbroadcast
aot::sum_backward
aot::transpose2d
aot::relu_backward
```

They are represented as `ExternalOp` instructions and execute through Dispatcher.

## Lowered representation

```text
LoweredProgram
├── ExternalOp
└── PointwiseFusion
```

`LoweringResult` exposes:

```text
normalized_graph
optimized_graph
program
fusion_groups
source_to_lowered_value
LoweringStats
```

Stats include:

```text
original_nodes
normalized_nodes
live_nodes_after_dce
constants_folded
optimized_nodes
fusion_groups
fused_nodes
external_nodes
lowered_instructions
```

`explain()` provides deterministic fusion/external-op inspection.

## LoweredExecutor

Execution policy:

```text
ExternalOp
   ↓
Dispatcher

PointwiseFusion
   ↓
NumPy reference expression
```

Pointwise fusion preserves CPU/EDU logical device identity.

The entire lowered execution runs under `no_grad` and is explicitly first-order. v0.14 does not construct a partially fused eager Autograd graph.

## AOT saved-value compatibility

Graph transformations rebuild Value ids, while v0.13 backward execution depends on an explicit saved-value ABI.

Therefore v0.14 composes Value-id mappings through the full pipeline:

```text
source Graph Value
      ↓
normalized Value
      ↓
post-DCE Value
      ↓
post-fold Value
      ↓
final lowered Value
```

API:

```python
lowered_id = result.mapped_value_id(source_id)
```

This keeps AOT saved forward Values usable after forward-graph lowering.

## Forward/backward unification

Because v0.13 made backward a normal Graph, the same lowering pipeline processes both:

```python
lower_graph(program.forward_graph)
lower_graph(program.backward_graph)
```

No separate backward optimizer is required.

## Correctness proof

v0.14 validates:

```text
GraphInterpreter(graph)
==
LoweredExecutor(lower_graph(graph))
```

for supported CPU/EDU programs, and also validates:

```text
v0.13 AOT backward-Graph gradients
==
v0.14 lowered backward-Graph gradients
```

Coverage includes:

- producer/user/use-count/liveness analysis;
- source-preserving deterministic normalization;
- dead branch elimination;
- safe constant folding;
- no folding of grad-requiring constants;
- pointwise chain fusion;
- shared-value branch fusion barrier;
- matmul/reduction barriers;
- CPU/EDU lowered execution;
- external-op Dispatcher execution;
- AOT forward saved-value remapping;
- AOT backward lowering and gradient equality;
- deterministic stats/explain output;
- input-count, mixed-device and malformed-Graph failures.

## Tests added

The full regression suite grew from 206 to **221 tests**.

## Important files

```text
mini_pytorch/compiler/analysis.py
mini_pytorch/compiler/lowering.py
mini_pytorch/compiler/__init__.py

tests/test_lowering.py
docs/LOWERING_FUSION.md
examples/17_lowering_fusion.py
```

Recommended reading order:

```text
compiler/ir.py
    ↓
compiler/analysis.py
    ↓
compiler/lowering.py
    ↓
test_lowering.py
    ↓
LOWERING_FUSION.md
    ↓
17_lowering_fusion.py
```

## Performance boundary

A reduction such as:

```text
3 high-level pointwise nodes
→
1 PointwiseFusion instruction
```

is a compiler-structure result, **not a speed claim**.

The v0.14 executor still uses Python/NumPy reference execution and materializes inspectable intermediate values. Runtime measurement and memory planning belong to v0.15.

## Non-goals

v0.14 does not implement:

- production TorchInductor;
- C++/CUDA/Triton code generation;
- GPU scheduling or autotuning;
- symbolic-shape lowering;
- arbitrary mutation/alias analysis;
- memory planning / buffer reuse;
- benchmark-backed performance optimization.

## Next version

v0.15 should study **Runtime / Performance Mechanics**: profiling, Value lifetime intervals, allocation/buffer reuse, preallocation, explicit execution plans, and measured eager-vs-graph-vs-lowered benchmarks.
