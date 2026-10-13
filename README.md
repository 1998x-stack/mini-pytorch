# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行的 Python / NumPy 代码，按真实 PyTorch 的架构依赖顺序，一层一层拆开核心机制。

**Latest integrated version:** `v0.11.0`

**Current milestone:** Graph IR / Eager-to-Graph Capture

**Next milestone:** `v0.12` Guards / Graph Cache / Recompile

mini-pytorch 不追求复制完整 PyTorch API，而是持续回答一条架构问题链：

```text
Tensor 是什么？
      ↓
Autograd 为什么能自动求导？
      ↓
Optimizer / Scheduler 如何组织训练动力学？
      ↓
Dataset / Sampler / DataLoader 为什么分层？
      ↓
Storage / stride / view 为什么重要？
      ↓
logical operator 为什么需要 Dispatcher？
      ↓
复杂 NN operator 如何建立在 Dispatcher 上？
      ↓
Device / Backend 如何与 Tensor 分离？
      ↓
AutogradEngine 如何管理 graph lifetime / hooks？
      ↓
完整训练状态如何 checkpoint 并确定性恢复？
      ↓
多个 replica 如何同步 gradient？
      ↓
eager execution 如何变成 Graph IR？
      ↓
Graph 在什么条件下可以安全复用？
```

## 1. 版本学习地图

| Version | 核心主题 | 最重要的问题 |
|---|---|---|
| `v0.1` | Tensor + Dynamic Autograd | `backward()` 如何自动求导？ |
| `v0.2` | Optimizer + LR Scheduler | 优化状态为什么独立于模型？ |
| `v0.3` | Dataset + Sampler + DataLoader + MNIST | 数据为什么分层？ |
| `v0.4` | Storage + stride + views + version counter | Tensor 为什么不等于 ndarray？ |
| `v0.5` | Operator Schema + Dispatcher + Redispatch | 一个 operator 为什么需要路由层？ |
| `v0.6` | CNN Operator Stack | 非平凡 NN op 如何复用 Dispatcher？ |
| `v0.7` | Device / Backend | 多 backend 为什么不能靠散落的 `if device`？ |
| `v0.8` | Advanced Autograd Engine | graph lifetime / hooks / retain_graph 如何工作？ |
| `v0.9` | Runtime / Checkpoint / RNG | 怎样暂停训练并确定性继续？ |
| `v0.10` | Distributed / DDP Concepts | 多个 replica 如何同步 gradient？ |
| `v0.11` | Graph IR / Eager Capture | eager operator history 如何表示成显式图？ |
| `v0.12` | Guards / Graph Cache / Recompile | 一个 Graph 什么时候仍然有效？ |

每个历史 version branch 都保留：

```text
README.md   → 本版本学习入口
version.md  → 与上一版本的精确 delta
Agents.md   → 本版本工程边界 / frozen contract
```

`main` 始终代表最新完成并验证的框架状态。

## 2. 当前完整架构

```text
                               Tensor
                   shape / stride / dtype / device
                                 │
                                 ▼
                               Storage
                 backend allocation + VersionCounter
                                 │
                                 ▼
                           logical operator
                                 │
                                 ▼
                             Dispatcher
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
           Autograd            NoGrad           CPU / EDU
              │                                      │
              └──────────── redispatch ──────────────┘
                                 │
                                 ▼
                         numerical kernel
                                 │
                                 ▼
                           Tensor / GradFn
                                 │
                                 ▼
                           AutogradEngine

Dataset → Sampler → BatchSampler → DataLoader

Module / Parameter → Optimizer → LR Scheduler
                              │
                        Training Runtime
                              │
                   RNG + CheckpointState

DistributedSampler
      ↓
DDP wrapper → Reducer → GradientBucket → ProcessGroup

Python eager function
      ↓
Dispatcher.dispatch
      ├── normal execution
      └── capture observer
              ↓
       Value / Node / Graph
              ↓
       GraphInterpreter
              ↓
          Dispatcher replay
```

## 3. v0.11 最新能力

### Explicit Graph IR

新增：

```text
mini_pytorch/compiler/
├── ir.py
├── capture.py
└── interpreter.py
```

IR 核心对象：

```text
Value
Node
Graph
```

Value kinds：

```text
input
constant
intermediate
```

### Capture

```python
from mini_pytorch.compiler import capture

captured = capture(
    lambda x, y: ((x * y) + x).relu().sum(),
    x,
    y,
)

print(captured.graph.format())
```

Capture 是普通 eager execution + observation，不是 symbolic-only fake run。

### Dispatcher observation boundary

Graph 只记录 logical operator：

```text
aten::mul.Tensor
```

不会把：

```text
Autograd → redispatch → CPU
```

重复记录成多节点。

### GraphInterpreter

```python
from mini_pytorch.compiler import GraphInterpreter

out = GraphInterpreter(captured.graph)(new_x, new_y)
```

Replay 仍然走：

```text
Dispatcher
→ runtime DispatchKeySet
→ Autograd / NoGrad
→ CPU / EDU
```

因此 Graph IR 不固化 trace 时的 backend/policy 决策。

### Forward / backward equivalence

v0.11 的中央证明：

```text
ordinary eager output == GraphInterpreter output
```

并验证：

```text
eager input gradients == replay input gradients
```

当前完整回归：

```text
176 tests
OK
```

### Explicit GraphBreak

如果 dynamic Tensor 来自尚未被 Dispatcher observer 覆盖的 eager path，例如当前 `exp()` direct Function path：

```python
y = x.exp()
z = y + x
```

capture 会显式抛出：

```text
GraphBreakError
```

而不是把 trace-time `y` 错误冻结成常量。

## 4. 推荐源码学习顺序

```text
v0.1  autograd.py → tensor.py → ops.py
v0.2  optim/optimizer.py → optim/* → lr_scheduler.py
v0.3  data/dataset.py → sampler.py → dataloader.py
v0.4  storage.py → tensor.py → docs/TENSOR_MEMORY.md
v0.5  dispatcher.py → operator_registry.py → docs/DISPATCHER.md
v0.6  kernels.py → nn_ops.py → nn/* → docs/CNN_OPERATORS.md
v0.7  device.py → backend.py → docs/DEVICE_BACKEND.md
v0.8  autograd_engine.py → docs/AUTOGRAD_ENGINE.md
v0.9  runtime/* → docs/TRAINING_RUNTIME.md
v0.10 distributed/* → docs/DISTRIBUTED_DDP.md
v0.11 compiler/ir.py → capture.py → dispatcher.py → interpreter.py → docs/GRAPH_IR.md
```

教学示例：

```bash
python examples/07_tensor_memory.py
python examples/08_dispatcher.py
python examples/09_cnn_mnist.py
python examples/10_device_backend.py
python examples/11_autograd_engine.py
python examples/12_checkpoint_resume.py
python examples/13_ddp_simulation.py
python examples/14_graph_capture.py
```

完整回归：

```bash
pip install -e .
python -m unittest discover -s tests -v
```

## 5. 当前明确非目标

截至 v0.11，项目刻意还没有实现：

- real CUDA / NCCL；
- true higher-order Autograd；
- real multi-process/network DDP；
- Python bytecode/frame tracing；
- symbolic control flow；
- shape/device/dtype guards；
- graph cache / automatic recompile；
- AOTAutograd；
- graph optimization/fusion；
- compiler lowering/codegen；
- Inductor/Triton。

这些会按架构依赖顺序进入后续版本。

## 6. 下一版：v0.12 Guards / Graph Cache / Recompile

v0.11 回答：

> Graph 是什么？

v0.12 要回答：

> 哪些 runtime assumptions 让这个 Graph 仍然有效？如果输入 shape/dtype/device/grad-mode 等发生变化，应该 cache hit、cache miss，还是重新 capture？

目标结构：

```text
Python callable
      ↓
runtime inputs
      ↓
GuardSet / Signature
      ↓
GraphCache lookup
   ┌───────┴────────┐
   ▼                ▼
hit               miss
   │                 │
GraphInterpreter   capture
   │                 ↓
   │             new Graph
   │                 ↓
   └──────────── cache entry
```

详细任务见 `Plan.md`。
