# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行的 Python / NumPy 代码，按真实 PyTorch 的架构依赖关系逐层拆开核心机制。

**Latest integrated version:** `v0.13.0`  
**Current milestone:** AOTAutograd-like Graph Differentiation  
**Next milestone:** `v0.14` Lowering / Fusion / Inductor-like Concepts

mini-pytorch 不追求 PyTorch API feature parity，而是回答一条逐步深入的架构问题链：

```text
Tensor / Autograd
      ↓
Optimizer / DataLoader
      ↓
Storage / stride / views
      ↓
Dispatcher / Backend
      ↓
Autograd Engine
      ↓
Training Runtime / DDP
      ↓
Graph IR
      ↓
Guards / Graph Cache
      ↓
AOT-style backward Graph
      ↓
Lowering / Fusion
      ↓
Runtime / Performance
```

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
| `v0.13` | AOTAutograd-like Graph Differentiation | **206** |

## 2. 当前完整架构

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
       kernels

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
GraphInterpreter replay
      ↓
AOT differentiation
  ┌───────────────┴───────────────┐
  ▼                               ▼
forward Graph                backward Graph
  │                               ▲
  └── user output + saved Values ─┘
                                  + grad_output
```

## 3. v0.13 核心

v0.13 第一次把 backward 从 eager Engine 的隐藏执行过程提升成显式 graph program：

```python
from mini_pytorch.compiler import aot_compile

program = aot_compile(fn, x_example, w_example)
result = program.forward(x, w)
dx, dw = program.backward(result.context)

print(program.forward_graph.format())
print(program.backward_graph.format())
```

当前 VJP 范围：

```text
add
mul
matmul (2-D)
relu
sum
```

关键机制：

- `DifferentiationSpec`；
- explicit saved forward Values；
- explicit grad-output ABI；
- backward Graph；
- broadcast `unbroadcast`；
- reduction backward；
- explicit shared-gradient accumulation；
- CPU/EDU Dispatcher replay；
- eager AutogradEngine 作为 reference oracle。

当前验证基线：

```text
206 / 206 tests passed
```

## 4. 推荐阅读 v0.13

```text
mini_pytorch/compiler/aot.py
      ↓
mini_pytorch/aot_kernels.py
      ↓
mini_pytorch/operator_registry.py
      ↓
mini_pytorch/compiler/interpreter.py
      ↓
tests/test_aot_autograd.py
      ↓
docs/AOT_AUTOGRAD.md
      ↓
examples/16_aot_autograd.py
```

## 5. Quick run

```bash
python examples/16_aot_autograd.py
python -m unittest discover -s tests -v
```

## 6. Compiler learning path

```text
v0.11  Graph IR
  ↓
v0.12  Guards / Cache / Recompile
  ↓
v0.13  AOT-style backward Graph
  ↓
v0.14  Lowering / Fusion
  ↓
v0.15  Runtime / Performance
  ↓
v1.0   Integrated Architecture
```

## 7. Next — v0.14

下一阶段回答：

> 当 forward 和 backward 都已经是显式 Graph 后，如何分析、规范化、消除无用计算、融合可融合节点，并执行一个更低层的程序表示？

v0.14 不会声称实现真实 Inductor/Triton，而会建立一个可读、可验证的 educational lowering pipeline。
