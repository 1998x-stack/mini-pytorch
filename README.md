# mini-pytorch v1.0 — Architecture-Driven PyTorch Learning Project

> 用可运行、可测试的 Python / NumPy 代码，把深度学习框架从 Tensor/Storage、Dispatcher、Autograd 一路拆到 DDP、Graph Compiler、AOT backward、Lowering 与 ExecutionPlan。

**Latest integrated release:** `v1.0.0`  
**Original learning sequence:** complete  
**Validated baseline:** **251 / 251 tests passed**  
**Integrated architecture example:** CI validated

mini-pytorch 不追求 PyTorch API feature parity，也不复制上游源码。它的目标是把真实框架里的关键责任边界拆成一系列小而完整的机制，然后在 v1.0 中把这些机制重新串成一个整体。

## 1. 完整学习版本地图

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

每个历史 branch 都保留其学习快照；`main` 始终代表最新集成、已验证的框架状态。

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
benchmark / profiling / lifetime analysis
   ↓
MemoryPlan
   ↓
ExecutionPlan
   ↓
PlannedExecutor
```

## 3. v1.0 的核心工作不是“再加功能”

v0.1–v0.15 已完成原计划里的主要机制。v1.0 做的是：

```text
architecture audit
      ↓
public API / naming audit
      ↓
cross-layer integration tests
      ↓
one end-to-end executable example
      ↓
PyTorch subsystem concept mapping
      ↓
documentation reconciliation
      ↓
release validation
```

因此 v1.0 的价值在于回答：

> 一个 Tensor 表达式，从内存布局、算子调度、反向传播、训练状态，到图捕获、AOT backward、lowering 和 runtime memory planning，分别由谁负责？这些层为什么不能混在一起？

## 4. 三个关键架构区分

### Tensor != Storage

```text
Storage
= flat allocation + backend identity + shared version counter

Tensor
= Storage reference + shape + stride + offset + autograd-facing state
```

这解释了 view / transpose / reshape / contiguous / in-place version safety 的不同语义。

### Dispatcher != kernel

```text
logical operator
      ↓
Dispatcher policy
      ↓
backend kernel
```

Autograd、NoGrad、backend selection、redispatch 属于调度层，不属于单个数值 kernel。

### Graph != LoweredProgram != ExecutionPlan

```text
Graph
= logical dataflow

LoweredProgram
= compiler 决定 what executes

ExecutionPlan
= runtime 决定 how temporary resources are assigned
```

这个区分把 compiler transform 与 runtime resource planning 明确分开。

## 5. Correctness stack

框架保留旧执行层作为新执行层的独立 reference：

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
lowered/planned backward gradients
```

新的优化层不替代 correctness oracle。

## 6. v1.0 新增的集成资产

### 完整架构审计

```text
docs/ARCHITECTURE_V1.md
```

逐层记录：

```text
owner / responsibility
inputs / outputs
allowed dependencies
forbidden dependencies
invariants
failure boundary
educational simplifications
```

### API 审计

```text
docs/API_AUDIT.md
```

结论是保留低层学习型 API，不为了表面模仿 PyTorch 做 breaking rewrite。

### PyTorch 概念映射

```text
docs/PYTORCH_MAPPING.md
```

使用：

```text
mini-pytorch concept
      ↕ responsibility analogy
real PyTorch subsystem family
```

明确不声明 source/API/performance parity。

### 跨层集成测试

```text
tests/test_v1_integration.py
```

验证：

```text
Tensor → Dispatcher → Backend → Autograd
Module → Optimizer → Scheduler → Checkpoint restore
Capture → Guard/Cache → AOT → Lowering → ExecutionPlan
Planned backward → eager gradient reference
```

### 最终集成示例

```bash
python examples/19_integrated_architecture.py
```

分成真实的不同阶段：

```text
1. eager training
2. runtime/checkpoint restore
3. graph cache specialization
4. eager gradient reference
5. AOT + lowering + planned runtime
```

DDP 不被强行塞进 compiler subgraph；它仍是 sibling training-synchronization concern。

## 7. Release metadata 统一

v1.0 引入：

```text
mini_pytorch/_version.py
```

作为运行时代码的单一版本源：

```text
mini_pytorch.__version__ = 1.0.0
checkpoint framework_version = 1.0.0
pyproject package version = 1.0.0
```

同时 CI 从只匹配 `v0.*` 改为匹配 `v*`，所以 v1.0 与未来明确创建的版本分支都可以获得 exact-head 验证。

## 8. v1.0 验证

Frozen v1.0 branch HEAD：

```text
7f6ef6ebb5357d7b15954e780455976fbff3f6b3
```

通过 GitHub Actions：

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
mini-pytorch-learning 1.0.0
```

Release gate：

```bash
python -m unittest discover -s tests -v
python examples/19_integrated_architecture.py
```

结果：

```text
251 tests / OK
integrated architecture example / PASS
```

## 9. 推荐阅读路径

完整底层路径：

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

配套阅读：

```text
docs/LEARNING_PATH.md
docs/ARCHITECTURE_V1.md
docs/PYTORCH_MAPPING.md
```

## 10. Quick start

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/19_integrated_architecture.py
```

也可以按历史版本逐层学习：

```text
v0.1 → v0.2 → ... → v0.15 → v1.0
```

## 11. Explicit non-goals

v1.0 不包括：

- real CUDA/Triton kernels/codegen；
- symbolic dynamic-shape compiler；
- production multiprocessing/prefetch/pinned-memory DataLoader；
- NCCL/network DDP；
- production caching allocator；
- whole-Python bytecode compiler；
- complete higher-order autodiff；
- PyTorch API/source/performance parity；
- benchmark marketing claims。

这些是明确边界，而不是隐藏的 v1.0 完成条件。

## 12. Historical branch rule

历史版本是学习快照。阅读某个版本时先看：

```text
Agents.md
version.md
README.md
```

冻结 branch 只接受历史边界内的文档修正、澄清测试或窄范围 bugfix。

未来开发必须从最新 `main` 开始。

## 13. After v1.0

**没有自动的 `v1.1` feature queue。**

如果继续项目，第一步不是创建下一分支，而是回答：

```text
缺少的 mental model 是什么？
为什么 v1.0 现有架构不能解释它？
它应该由哪一层负责？
最小可读实现是什么？
明确的 non-goals 是什么？
```

只有这些问题形成新的独立 roadmap 后，才应该创建后续版本。
