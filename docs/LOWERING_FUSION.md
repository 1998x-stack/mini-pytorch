# Lowering / Fusion Architecture — mini-pytorch v0.14

## 1. Why lowering exists

v0.11-v0.13 established an explicit compiler-visible program:

```text
eager execution
   ↓
Graph IR
   ↓
guards/cache
   ↓
forward Graph + backward Graph
```

Those Graphs are correct and replayable, but they still use high-level logical operators. v0.14 asks a different question:

> How can a correct Graph be analyzed and transformed into a lower-level execution plan without changing its semantics?

The answer is deliberately educational rather than production-grade:

```text
Graph
  ↓
normalize
  ↓
analyze
  ↓
DCE
  ↓
constant fold
  ↓
fusion planning
  ↓
LoweredProgram
  ↓
LoweredExecutor
```

`GraphInterpreter` remains the semantic reference.

---

## 2. Policy vs mechanism

v0.14 preserves the project-wide separation:

```text
compiler transformation policy
          !=
numerical execution mechanism
```

Examples:

- `plan_fusions()` decides which nodes may form a fusion group;
- `PointwiseFusion` is an explicit lower-level instruction;
- `LoweredExecutor` implements a readable NumPy reference mechanism;
- `ExternalOp` continues to execute through Dispatcher.

The project does **not** claim that the NumPy fusion executor is a generated optimized kernel.

---

## 3. GraphAnalysis

`analyze_graph(graph)` produces inspectable data for every Value:

```text
ValueAnalysis
├── producer_node_id
├── user_node_ids
├── use_count
├── is_graph_output
└── is_live
```

### Why use_count counts occurrences

Consider:

```python
shared = x * y
out = shared + shared
```

`shared` has one user node but two uses.

For fusion safety this matters:

```text
use_count = 2
```

therefore the producer should not be treated as a single-consumer linear chain.

### Liveness

Liveness is computed by reverse reachability:

```text
graph outputs
      ↓
producer nodes
      ↓
node dependencies
      ↓
inputs/constants
```

This is enough for the pure SSA-like Graph model used by mini-pytorch v0.14.

---

## 4. Normalization

`normalize_graph(graph)` rebuilds a deterministic clone.

Important invariant:

```text
source graph before pass
==
source graph after pass
```

The pass does not mutate the historical Graph.

Normalization preserves:

- graph input order;
- graph output structure;
- node order;
- constant payloads;
- Tensor metadata;
- logical operator names and kwargs.

Values are remapped deterministically into the rebuilt Graph.

---

## 5. Dead Code Elimination

Captured eager execution can contain computations irrelevant to the returned output:

```python
def fn(x):
    dead = x * x
    live = x + x
    return live
```

The captured graph contains both nodes, but reverse reachability from the output reaches only `live`.

DCE therefore rebuilds:

```text
input x
   ↓
add(x, x)
   ↓
return
```

### Side-effect boundary

This is safe only because the educational compiler currently treats its captured logical tensor operators as pure.

v0.14 does not claim general Python or mutation side-effect analysis.

---

## 6. Safe constant folding

Initial foldable set:

```text
aten::add.Tensor
aten::mul.Tensor
aten::relu.default
```

A node may fold only when:

1. every referenced Tensor Value is a graph constant;
2. constants share one logical device;
3. none of those constants requires grad;
4. the output metadata does not require grad;
5. the operator is in the documented pure subset.

The fold is evaluated through normal Dispatcher semantics under `no_grad`, then snapshotted as a new `TensorConstant`.

### Why grad-requiring constants are not folded

A closure Tensor with `requires_grad=True` may participate in gradient semantics. Replacing its expression with an inert constant would change the meaning of the program.

Therefore v0.14 chooses the conservative rule:

```text
requires_grad constant
→ do not fold
```

---

## 7. Fusion eligibility

Initial pointwise set:

```text
add
mul
relu
```

A chain may extend only when the previous output:

```text
has use_count == 1
AND is not a graph output
AND is consumed by the next pointwise node
```

This gives a simple linear-chain fusion model.

### Safe chain

```text
mul
 ↓ one use
add
 ↓ one use
relu
```

becomes one fusion group.

### Branch barrier

```text
        ┌→ add
mul ────┤
        └→ mul
```

The first output has multiple uses, so it cannot be swallowed into one linear group.

This is intentionally conservative.

---

## 8. Explicit FusionGroup data

A fusion decision is stored as data:

```text
FusionGroup
├── id
├── node_ids
├── input_value_ids
└── output_value_ids
```

This matters because later runtime or code-generation stages should consume explicit compiler decisions rather than re-derive them from adjacency.

---

## 9. Fusion barriers

v0.14 initially leaves these outside pointwise fusion:

```text
matmul
sum
conv2d
max_pool2d
AOT unbroadcast
AOT sum backward
AOT transpose2d
AOT relu backward
```

They become `ExternalOp` instructions.

This is not a statement that real compilers can never fuse them. It is a scope choice that keeps the first lowering model readable.

---

## 10. LoweredProgram

The lower-level representation is intentionally tiny:

```text
LoweredProgram
├── ExternalOp
└── PointwiseFusion
```

Example high-level Graph:

```text
%2 = mul(%0, %1)
%3 = add(%2, %0)
%4 = relu(%3)
%5 = sum(%4)
```

may lower to:

```text
[0] PointwiseFusion(mul -> add -> relu)
[1] ExternalOp(sum)
```

This is already a meaningful compiler distinction even without code generation.

---

## 11. LoweredExecutor

### ExternalOp

```text
ExternalOp
   ↓
Dispatcher
   ↓
normal backend kernel
```

### PointwiseFusion

```text
PointwiseFusion
   ↓
reference NumPy expression evaluation
   ↓
Tensor result on same logical device
```

The reference executor still materializes intermediate Tensor values in its environment for inspectability and AOT saved-value access. Therefore it is a semantic model of fusion, not yet a memory-optimized kernel runtime.

---

## 12. First-order execution boundary

The entire `LoweredExecutor` run occurs under `no_grad`.

Why?

Without this rule, a program containing:

```text
fused pointwise region
+
ExternalOp
```

could accidentally build only part of an eager Autograd graph. That hybrid would have confusing and incorrect semantics.

Instead v0.14 says:

```text
lowered forward = numerical program
lowered backward = explicit numerical backward Graph
```

Gradient correctness comes from v0.13's explicit backward Graph.

---

## 13. Source-to-lowered Value mapping

Transform passes rebuild Value ids. AOT has an important ABI:

```text
forward saved_value_ids
```

Therefore `LoweringResult` stores a composed mapping:

```text
source Value id
      ↓
normalized id
      ↓
DCE id
      ↓
constant-fold id
      ↓
final lowered Graph id
```

API:

```python
lowered_id = result.mapped_value_id(source_id)
```

This allows an AOT forward Graph to be transformed without losing the saved-value interface required by its backward Graph.

---

## 14. Forward and backward use the same pipeline

v0.13 made backward a normal `Graph`.

That means v0.14 needs no separate backward optimizer:

```python
forward_lowered = lower_graph(program.forward_graph)
backward_lowered = lower_graph(program.backward_graph)
```

The same analysis, DCE, folding and fusion machinery applies to both.

This is one of the main architectural payoffs of explicit graph differentiation.

---

## 15. Correctness triangle

For forward execution:

```text
eager/reference values
        ==
GraphInterpreter
        ==
LoweredExecutor
```

For gradients:

```text
eager AutogradEngine
        ==
v0.13 explicit backward Graph
        ==
v0.14 lowered backward Graph
```

This layered reference structure prevents optimization passes from becoming their own correctness oracle.

---

## 16. Stats and explainability

`lower_graph()` returns inspectable statistics:

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

And:

```python
print(result.explain())
```

shows fusion groups and remaining external operations.

The output is deterministic for the same Graph.

---

## 17. What v0.14 does not prove

A smaller instruction count is not a benchmark.

For example:

```text
3 pointwise Graph nodes
→ 1 PointwiseFusion instruction
```

does not by itself prove faster execution, because the reference implementation still runs Python and NumPy operations.

Actual questions such as:

- allocation count;
- intermediate lifetime;
- dispatch overhead;
- buffer reuse;
- execution timing;
- preallocation;

belong to v0.15.

---

## 18. Relationship to real PyTorch / Inductor

Conceptually:

```text
mini-pytorch Graph
      ↓
analysis / transforms
      ↓
LoweredProgram
      ↓
reference executor
```

helps build intuition for the broader compiler pipeline:

```text
PyTorch graph
      ↓
decomposition / analysis
      ↓
scheduling / fusion
      ↓
lower-level IR
      ↓
code generation / runtime
```

The implementation is intentionally much smaller and does not claim TorchInductor parity.

---

## 19. Reading order

```text
compiler/ir.py
   ↓
compiler/analysis.py
   ↓
compiler/lowering.py
   ↓
tests/test_lowering.py
   ↓
examples/17_lowering_fusion.py
```
