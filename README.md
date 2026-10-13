# mini-pytorch v0.14 — Lowering / Fusion / Inductor-like Concepts

> 本版本专门研究：当 forward / backward 都已经是显式 Graph 后，如何把高层 logical Graph 转换成可分析、可优化、可融合、可执行的 lower-level program。

**Branch:** `v0.14`  
**Previous:** `v0.13` AOTAutograd-like Graph Differentiation  
**Package:** `0.14.0`

## 1. 核心变化

v0.13 已经得到：

```text
forward Graph
+
backward Graph
```

v0.14 增加：

```text
Graph
  ↓
normalize_graph
  ↓
analyze_graph
  ↓
DCE
  ↓
safe constant folding
  ↓
fusion planning
  ↓
LoweredProgram
  ↓
LoweredExecutor
```

`GraphInterpreter` 继续作为 reference executor。

## 2. 新增分析层

```python
from mini_pytorch.compiler import analyze_graph

analysis = analyze_graph(graph)
print(analysis.format())
```

每个 Value 都有显式信息：

```text
producer_node_id
user_node_ids
use_count
is_graph_output
is_live
```

这些信息是 DCE 和 fusion 的依赖，而不是在 pass 内临时猜测。

## 3. Graph normalization

```python
normalized = normalize_graph(graph)
```

特点：

- 不修改 source Graph；
- deterministic Value / Node rebuilding；
- input/output ordering 保持；
- constant payload / metadata 保持；
- before / after 都执行 validation。

## 4. Dead Code Elimination

```text
Graph outputs
     ↓ reverse reachability
live Values / Nodes
     ↓
rebuilt Graph
```

例如：

```python
def fn(x):
    dead = x * x
    live = x + x
    return live
```

DCE 后 `dead` multiply 被移除。

v0.14 只对当前 educational Graph 中被视作 pure 的节点做这一推理，不声称拥有通用 side-effect analysis。

## 5. Safe constant folding

首批 foldable logical ops：

```text
add
mul
relu
```

只在所有 Tensor inputs 都是 captured constants、同 device、且不 requires_grad 时折叠。

例如：

```text
constant A
constant B
   ↓
A * B
   ↓
relu
```

可在 lowering 时变成一个新的 `TensorConstant`。

Grad-requiring constants 不折叠，避免改变梯度语义。

## 6. Pointwise fusion

首批 fusible ops：

```text
aten::add.Tensor
aten::mul.Tensor
aten::relu.default
```

例如：

```text
x ──┐
    mul ── add ── relu
 y ─┘      ▲
           x
```

可以形成：

```text
PointwiseFusion
  nodes = [mul, add, relu]
```

但 fusion 不是简单把相邻节点贴标签。中间 Value 必须满足真实 dependency/use-count 条件。

如果：

```text
shared
 ├── branch A
 └── branch B
```

则 producer 不会被错误吞入一个单链 fusion group。

## 7. Fusion barriers

当前明确作为 barrier：

```text
matmul
sum
conv / pool
aot::unbroadcast
aot::sum_backward
aot::transpose2d
aot::relu_backward
```

它们进入：

```text
ExternalOp
```

并继续通过正常 Dispatcher 执行。

## 8. Lowered representation

```text
LoweredProgram
├── ExternalOp
└── PointwiseFusion
```

查看：

```python
result = lower_graph(graph)
print(result.program.format())
print(result.stats().format())
print(result.explain())
```

可观察 metrics 包括：

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

注意：

```text
fewer nodes != automatically faster
```

性能测量属于 v0.15。

## 9. LoweredExecutor

```python
lowered = lower_graph(graph)
out = LoweredExecutor(lowered.program)(x, y)
```

执行策略：

```text
ExternalOp
   ↓
Dispatcher

PointwiseFusion
   ↓
NumPy reference expression
```

fusion executor 会保留 CPU / EDU logical device identity，但不会生成 C++/CUDA/Triton kernel。

整个 lowered run 是 first-order numerical execution，不会构造一个“部分 fused、部分 eager”的 Autograd graph。

## 10. 为什么 v0.13 很重要

因为 backward 在 v0.13 已成为显式 Graph，所以 v0.14 的同一 lowering pipeline 可以直接处理：

```text
AOTProgram.forward_graph
AOTProgram.backward_graph
```

核心 reference proof：

```text
GraphInterpreter(forward)
==
LoweredExecutor(lowered forward)
```

以及：

```text
AOT backward Graph reference gradients
==
LoweredExecutor(lowered backward) gradients
```

## 11. 重要文件

```text
mini_pytorch/compiler/analysis.py
mini_pytorch/compiler/lowering.py

tests/test_lowering.py
docs/LOWERING_FUSION.md
examples/17_lowering_fusion.py
```

推荐阅读：

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

## 12. Quick run

```bash
python examples/17_lowering_fusion.py
python -m unittest discover -s tests -v
```

## 13. Non-goals

v0.14 不实现：

- production TorchInductor；
- Triton / CUDA / C++ code generation；
- GPU scheduling / autotuning；
- symbolic shapes；
- arbitrary alias/mutation analysis；
- memory planning / buffer reuse；
- 未经 benchmark 的性能结论。

## 14. Next

`v0.15` 将研究 **Runtime / Performance Mechanics**：profiling、Value lifetime、buffer reuse、preallocation、execution plan 和真实 benchmark。
