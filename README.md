# mini-pytorch v1.0 — Integrated Architecture

> 用可运行、可测试的 Python / NumPy 代码，把一个 Tensor 表达式从 Storage、Dispatcher、Autograd 一路追踪到 Graph Compiler、AOT backward、Lowering、ExecutionPlan 与 PlannedExecutor。

**Release:** `1.0.0`  
**Branch:** `v1.0`  
**Status:** release candidate  
**Validated baseline:** **251 / 251 tests passed**  
**Integrated example:** CI validated

mini-pytorch 不是 PyTorch API clone，也不是性能复刻。它是一套按架构依赖逐版本构造的学习框架：每个版本只引入一个核心 mental model，v1.0 则把全部 0.x 机制收敛成一个可解释、可运行、可验证的整体。

## 1. 完整版本学习地图

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
| `v0.13` | AOT-style Graph Differentiation | 206 |
| `v0.14` | Lowering / Fusion / Inductor-like Concepts | 221 |
| `v0.15` | Runtime / Performance Mechanics | 247 |
| `v1.0` | Integrated Architecture | **251** |

## 2. v1.0 完整架构

```text
Python API
   ↓
Tensor / Storage / Device
   ↓
logical operator
   ↓
Dispatcher
   ├── Autograd / NoGrad policy
   └── CPU / EDU backend
          ↓
        kernels
          ↓
Tensor + GradFn
   ↓
AutogradEngine

Module → Parameter → Optimizer → Scheduler
Dataset → Sampler → BatchSampler → DataLoader
Runtime → RNG → Checkpoint / TrainingState
DDP → Reducer → GradientBucket → ProcessGroup

Python Tensor function
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
analysis / DCE / constant folding / fusion
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

## 3. v1.0 做了什么

v1.0 不增加新的大子系统，而是完成六件事：

```text
architecture audit
      ↓
public API / naming audit
      ↓
cross-layer integration tests
      ↓
one integrated executable example
      ↓
PyTorch concept mapping
      ↓
documentation + release consistency
```

### 新增核心文档

```text
docs/ARCHITECTURE_V1.md
    完整 ownership / dependency / failure-boundary 审计

docs/API_AUDIT.md
    公共 API 与命名审计

docs/PYTORCH_MAPPING.md
    mini-pytorch ↔ PyTorch subsystem concept map
```

### 新增跨层验证

```text
tests/test_v1_integration.py
```

覆盖：

```text
Tensor → Dispatcher → Backend → Autograd
Module → Optimizer → Scheduler → Checkpoint restore
Capture → Guards/Cache → AOT → Lowering → ExecutionPlan
Planned forward/backward → eager output/gradient reference
```

### 新增最终集成示例

```bash
python examples/19_integrated_architecture.py
```

它明确分成：

```text
phase 1  eager training
phase 2  checkpoint / runtime state restore
phase 3  graph capture + guards/cache
phase 4  eager gradient reference
phase 5  AOT + lowering + planned runtime
```

CI 已验证该示例完整运行，并验证：

```text
planned output == eager output
planned gradients == eager gradients
```

## 4. 三个最重要的架构区分

### Tensor != Storage

```text
Storage
= flat allocation + backend identity + shared version counter

Tensor
= Storage reference + shape + stride + offset + autograd-facing state
```

因此 transpose/view 可以共享 Storage，而 reshape/contiguous 在必要时可以分配新 Storage。

### Dispatcher != kernel

```text
logical operator
      ↓
Dispatcher policy
      ↓
backend kernel
```

Autograd、NoGrad、backend 选择、redispatch 是调度层责任，不属于单个 CPU 数值 kernel。

### Graph != LoweredProgram != ExecutionPlan

```text
Graph
= logical dataflow

LoweredProgram
= compiler 决定“执行什么”

ExecutionPlan
= runtime 决定“临时资源怎样分配”
```

这是从 eager framework 进入 compiler/runtime 架构时最关键的分层。

## 5. Correctness stack

mini-pytorch 刻意保留旧层作为新层的 reference：

```text
Eager Tensor execution
      ==
GraphInterpreter
      ==
LoweredExecutor
      ==
PlannedExecutor
```

在支持的 AOT 子集上：

```text
eager AutogradEngine gradients
      ==
explicit AOT backward Graph gradients
      ==
lowered backward gradients
      ==
planned backward gradients
```

优化层不会因为“更高级”就取代 reference layer。

## 6. v1.0 release consistency

v1.0 把版本信息收敛到：

```text
mini_pytorch/_version.py
__version__ = "1.0.0"
```

并让：

```text
mini_pytorch.__version__
checkpoint framework_version
pyproject package version
```

保持一致。

GitHub Actions 也从只监听 `v0.*` 扩展为：

```text
main
v*
```

因此 v1.0 能获得真正的 exact-head CI。

## 7. 当前验证结果

release-candidate framework state：

```text
commit d49b1872274f790303d719034074b75c270a18c2
```

GitHub Actions 环境：

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

结果：

```text
Ran 251 tests
OK

examples/19_integrated_architecture.py
PASS
```

## 8. 推荐阅读路径

如果想从底层一路读完：

```text
device.py / backend.py
      ↓
storage.py / tensor.py
      ↓
dispatcher.py / operator_registry.py / kernels.py
      ↓
autograd.py / autograd_engine.py
      ↓
nn / optim / data
      ↓
runtime / distributed
      ↓
compiler/ir.py / capture.py
      ↓
signature.py / guards.py / cache.py / compiled.py
      ↓
aot.py
      ↓
analysis.py / lowering.py
      ↓
runtime_analysis.py / execution_plan.py
```

配合：

```text
docs/LEARNING_PATH.md
docs/ARCHITECTURE_V1.md
docs/PYTORCH_MAPPING.md
```

## 9. Quick validation

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/19_integrated_architecture.py
```

## 10. PyTorch mapping 的正确用法

`docs/PYTORCH_MAPPING.md` 使用的是：

```text
mini-pytorch educational concept
      ↕ conceptual responsibility
real PyTorch subsystem family
```

而不是：

```text
同名类
=
同一份实现
```

例如：

```text
mini Dispatcher         ↔ PyTorch dispatcher concept family
mini AOT differentiation ↔ AOTDispatcher/AOTAutograd concept family
mini lowering/fusion     ↔ Inductor-like compiler concepts
ExecutionPlan            ↔ general compiler/runtime memory-planning ideas
```

项目明确不声明源码、API、性能或 edge-case parity。

## 11. Explicit non-goals

v1.0 不包括：

- real CUDA/Triton kernel/codegen；
- symbolic dynamic-shape compiler；
- production multiprocessing/prefetch DataLoader；
- NCCL/network DDP；
- production caching allocator；
- whole-Python bytecode compiler；
- complete higher-order autodiff；
- PyTorch feature parity；
- 性能营销式 benchmark 结论。

这些不是 v1.0 “没做完”，而是本项目原始学习目标之外的边界。

## 12. Historical branches

每个版本 branch 都是一个学习快照：

```text
v0.1 ... v0.15
v1.0
```

阅读历史版本时先看：

```text
Agents.md
version.md
README.md
```

冻结版本不作为未来开发基线；未来若要继续扩展，必须从最新 `main` 开始，并先定义新的 mental model 和独立 roadmap。

## 13. v1.0 的完成标准

v1.0 的最终目标不是“多一个功能”，而是让读者可以回答：

> 一个 Tensor 表达式，从内存布局、算子调度、反向传播、训练状态，到图捕获、AOT backward、lowering、runtime memory planning，分别由谁负责？为什么必须这样分层？

`docs/ARCHITECTURE_V1.md` 和 `examples/19_integrated_architecture.py` 是回答这个问题的两个入口：一个负责完整解释，一个负责可执行证明。
