# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行、可测试的 Python / NumPy 代码，从 Tensor/Autograd 一路拆到 Dispatcher、Backend、DDP、Graph Compiler、AOT backward、Lowering 与 Runtime Planning。

**Latest integrated version:** `v0.15.0`  
**Current completed milestone:** Runtime / Performance Mechanics  
**Next milestone:** `v1.0` Integrated Architecture

mini-pytorch 的目标不是 PyTorch API feature parity，也不是性能复刻，而是把真实深度学习框架的关键架构边界拆成可以逐版本阅读的最小机制。

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
| `v0.14` | Lowering / Fusion / Inductor-like Concepts | 221 |
| `v0.15` | Runtime / Performance Mechanics | **247** |

## 2. 当前完整架构

```text
Python API
   ↓
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

Tensor / GradFn
   ↓
AutogradEngine

Module → Parameter → Optimizer → Scheduler
Dataset → Sampler → DataLoader
Runtime → RNG → Checkpoint
DDP → Reducer → GradientBucket → ProcessGroup

Python eager function
   ↓
Dispatcher-level Graph capture
   ↓
Value / Node / Graph
   ↓
InputSignature / GuardSet / GraphCache
   ↓
AOT differentiation
   ├── forward Graph
   └── backward Graph
         ↑
 saved Values + grad_output
   ↓
Graph analysis
   ↓
DCE / constant folding / fusion
   ↓
LoweredProgram
   ├── ExternalOp
   └── PointwiseFusion
   ↓
profiling / lifetime analysis
   ↓
MemoryPlan
   ↓
ExecutionPlan
   ↓
PlannedExecutor
```

## 3. v0.15 — compiler/runtime handoff

v0.15 adds the final major mechanism before v1.0 consolidation:

```python
from mini_pytorch.compiler import (
    build_execution_plan,
    lower_graph,
    PlannedExecutor,
)

lowered = lower_graph(graph)
plan = build_execution_plan(lowered.program)
out = PlannedExecutor(plan)(x, y)
```

The split is intentional:

```text
LoweredProgram
= what operations execute

ExecutionPlan
= how runtime resources are assigned
```

v0.15 includes:

- reproducible benchmark primitives;
- per-instruction profiling;
- explicit Value lifetime analysis;
- conservative pointwise temporary buffer reuse;
- `BufferSpec / BufferInterval / BufferSlot / MemoryPlan`;
- `ExecutionPlan`;
- `PlannedExecutor`;
- allocation/reuse/peak-live metrics;
- AOT saved-value retention through planned forward/backward execution;
- capture object-identity lifetime robustness.

Validation:

```text
247 / 247 tests passed
```

## 4. Correctness stack

The project now has multiple independent reference levels:

```text
Eager Tensor execution
      ↓ reference
GraphInterpreter
      ↓ reference
LoweredExecutor
      ↓ reference
PlannedExecutor
```

For gradients:

```text
eager AutogradEngine
      ==
AOT explicit backward Graph
      ==
lowered backward Graph
      ==
planned backward Graph
```

Optimization layers do not replace the older reference layers; they are checked against them.

## 5. Performance philosophy

mini-pytorch does not infer speed from representation changes.

```text
fusion count ↓
allocation count ↓
instruction count ↓
```

are structural facts, not latency guarantees.

Performance discussions require:

```text
warmup
repeated measurements
median / distribution
same workload
separate planning and steady-state timing
allocation metrics reported separately from wall clock
```

Run:

```bash
python examples/18_runtime_performance.py
```

The example compares:

```text
Eager
GraphInterpreter
LoweredExecutor
PlannedExecutor
```

without asserting which one must be fastest.

## 6. Recommended compiler/runtime reading path

```text
compiler/ir.py
   ↓
compiler/capture.py
   ↓
compiler/guards.py + cache.py + compiled.py
   ↓
compiler/aot.py
   ↓
compiler/analysis.py
   ↓
compiler/lowering.py
   ↓
compiler/benchmark.py
   ↓
compiler/profiler.py
   ↓
compiler/runtime_analysis.py
   ↓
compiler/execution_plan.py
```

## 7. Full quick validation

```bash
python -m unittest discover -s tests -v
```

Current integrated baseline:

```text
247 tests
OK
```

## 8. Historical branch rule

Each `v0.x` branch is a learning snapshot. Read that branch's:

```text
Agents.md
version.md
README.md
```

before studying the code. Future development starts from latest `main`, not from a frozen historical branch.

## 9. Next — v1.0 Integrated Architecture

The project now has all major mechanisms planned for the 0.x learning sequence. v1.0 should **consolidate rather than expand**.

Target work:

```text
architecture audit
      ↓
API / naming consistency
      ↓
end-to-end integrated example
      ↓
PyTorch subsystem mapping
      ↓
documentation consistency
      ↓
final regression / release snapshot
```

v1.0 is successful when a reader can start from a Tensor expression and trace, in one coherent map, how it reaches:

```text
Storage
Dispatcher
Backend
Autograd
Module / Optimizer / Data
Runtime / Checkpoint / DDP
Graph capture / Guards
AOT backward
Lowering / Fusion
ExecutionPlan / Buffer reuse
```

without needing a new major subsystem.
