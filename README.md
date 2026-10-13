# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行的 Python / NumPy 代码，沿着真实 PyTorch 的架构依赖逐层拆解 Tensor、Autograd、Dispatcher、Runtime、Distributed 与 Compiler。

**Latest integrated version:** `v0.14.0`  
**Current milestone:** Lowering / Fusion / Inductor-like Concepts  
**Next milestone:** `v0.15` Runtime / Performance Mechanics

mini-pytorch 不追求 PyTorch API feature parity，也不复制上游源码。项目重点是建立一条可运行、可测试、可逐版本回看的机制学习链。

## 1. 版本学习地图

| Version | 核心主题 | 验证基线 |
|---|---|---:|
| `v0.1` | Tensor / Dynamic Autograd | historical |
| `v0.2` | Optimizer state / LR Scheduler | historical |
| `v0.3` | Dataset / Sampler / DataLoader / MNIST | historical |
| `v0.4` | Storage / stride / views / version safety | historical |
| `v0.5` | Operator Registry / Dispatcher / Redispatch | 80 |
| `v0.6` | CNN Operator Stack | 90 |
| `v0.7` | Device / Backend | 103 |
| `v0.8` | Advanced Autograd Engine | 123 |
| `v0.9` | Training Runtime / Checkpoint / RNG | 140 |
| `v0.10` | Distributed / DDP Concepts | 159 |
| `v0.11` | Graph IR / Eager Capture | 176 |
| `v0.12` | Guards / Graph Cache / Recompile | 192 |
| `v0.13` | AOTAutograd-like Graph Differentiation | 206 |
| `v0.14` | Lowering / Fusion / Inductor-like Concepts | **221** |

## 2. 当前完整依赖链

```text
Tensor
  ↓
Storage + shape + stride + offset + Device
  ↓
logical operator
  ↓
Dispatcher
  ├── Autograd / NoGrad policy
  └── CPU / EDU backend
          ↓
        kernel

Tensor / GradFn → AutogradEngine

Dataset → Sampler → BatchSampler → DataLoader
Module → Parameter → Optimizer → Scheduler
Runtime → RNG → Checkpoint
DDP → Reducer → GradientBucket → ProcessGroup

Python eager function
      ↓
Dispatcher capture
      ↓
Value / Node / Graph
      ↓
InputSignature / GuardSet / GraphCache
      ↓
GraphInterpreter
      ↓
AOT differentiation
  ┌──────────────┴──────────────┐
  ▼                             ▼
forward Graph              backward Graph
  │                             ▲
  └── output + saved Values ────┘
                                + grad_output

forward/backward Graph
      ↓
normalize + analyze
      ↓
DCE + safe constant folding
      ↓
use-aware fusion planning
      ↓
LoweredProgram
      ↓
LoweredExecutor
```

## 3. v0.14 核心

v0.14 第一次把“正确的 Graph”推进成“经过显式 compiler passes 的 lower-level execution plan”。

```python
from mini_pytorch.compiler import lower_graph, LoweredExecutor

result = lower_graph(graph)
print(result.optimized_graph.format())
print(result.program.format())
print(result.explain())

out = LoweredExecutor(result.program)(x, y)
```

### Graph analysis

```text
ValueAnalysis
├── producer_node_id
├── user_node_ids
├── use_count
├── is_graph_output
└── is_live
```

`use_count` 按真实 occurrence 计数，因此：

```python
out = shared + shared
```

会得到 `use_count == 2`，避免把共享值错误吞进单消费者 fusion chain。

### Transform pipeline

```text
normalize_graph
      ↓
eliminate_dead_code
      ↓
fold_constants
      ↓
plan_fusions
      ↓
lower_graph
```

### Pointwise fusion

首批 fusible logical ops：

```text
add
mul
relu
```

首批 barriers：

```text
matmul
sum
conv / pool
AOT backward helpers
```

Lowered IR：

```text
LoweredProgram
├── ExternalOp
└── PointwiseFusion
```

`ExternalOp` 继续通过 Dispatcher；`PointwiseFusion` 使用可读的 NumPy reference expression 执行。

## 4. v0.14 最重要的架构连接

v0.13 已把 backward 变成普通 Graph，因此 v0.14 不需要第二套 backward optimizer：

```python
forward_lowered = lower_graph(program.forward_graph)
backward_lowered = lower_graph(program.backward_graph)
```

同时 `LoweringResult.source_to_lowered_value` 保留 source Value 到 transformed Value 的映射，使 v0.13 的 saved-value ABI 在 forward lowering 后仍然成立。

核心 correctness chain：

```text
GraphInterpreter
      ==
LoweredExecutor
```

以及：

```text
eager AutogradEngine gradients
      ==
v0.13 explicit backward Graph
      ==
v0.14 lowered backward Graph
```

## 5. 性能边界

v0.14 **不声称性能提升**。

例如：

```text
mul + add + relu
→ one PointwiseFusion instruction
```

只说明 compiler representation 已经融合，不意味着当前 Python/NumPy reference executor 一定更快。

真正的：

```text
profiling
Value lifetime
allocation count
buffer reuse
preallocation
peak live memory
benchmark
```

统一留到 v0.15。

## 6. 推荐阅读 v0.14

```text
mini_pytorch/compiler/ir.py
      ↓
mini_pytorch/compiler/analysis.py
      ↓
mini_pytorch/compiler/lowering.py
      ↓
tests/test_lowering.py
      ↓
docs/LOWERING_FUSION.md
      ↓
examples/17_lowering_fusion.py
```

## 7. Quick run

```bash
python examples/17_lowering_fusion.py
python -m unittest discover -s tests -v
```

当前验证基线：

```text
221 / 221 tests passed
```

## 8. Compiler learning path

```text
v0.11  Graph IR
   ↓
v0.12  Guards / Cache / Recompile
   ↓
v0.13  AOT-style backward Graph
   ↓
v0.14  Lowering / Fusion
   ↓
v0.15  Runtime / Performance Mechanics
   ↓
v1.0   Integrated Architecture
```

## 9. Next — v0.15

下一阶段回答：

> 当 LoweredProgram 已经正确后，runtime 如何测量每条 instruction、推导 Value 生命周期、减少 allocation、复用 buffer，并用可重复 benchmark 判断优化到底有没有收益？

v0.15 的原则是 **measurement before optimization**：没有测量结果，就不做速度结论。
