# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行的 Python / NumPy 代码，按真实 PyTorch 的架构依赖顺序，一层一层拆开核心机制。

**Latest integrated version:** `v0.12.0`

**Current milestone:** Guards / Graph Cache / Recompile

**Next milestone:** `v0.13` AOTAutograd-like Graph Differentiation

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
      ↓
backward 能否也变成显式 Graph？
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
| `v0.12` | Guards / Graph Cache / Recompile | 一个 Graph 什么时候还能安全复用？ |

## 2. 当前完整架构

```text
                             Tensor
                shape / stride / dtype / device
                               │
                               ▼
                             Storage
                  backend-owned allocation
                     + VersionCounter
                               │
                               ▼
                        logical operator
                               │
                               ▼
                           Dispatcher
                    policy keys + backend keys
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
          Autograd           NoGrad        CPU / EDU backend
             │                                   │
             └────────── redispatch ─────────────┘
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
                           ▼
                    Training Runtime
                  RNG + CheckpointState

DistributedSampler
      ↓
DDP → Reducer → GradientBucket → ProcessGroup

Python eager function
      ↓
Dispatcher.dispatch
      ├── normal eager execution
      └── capture observer
              ↓
       Value / Node / Graph
              ↓
       InputSignature
              ↓
          GuardSet
              ↓
         GraphCache
       ┌──────┴──────┐
       ▼             ▼
      HIT           MISS
       │             │
GraphInterpreter   capture
       │             │
       └────── Dispatcher replay
```

## 3. v0.12 当前重点

v0.12 将 v0.11 的“一个 execution → 一个 Graph”推进为：

```text
one Python callable
      ↓
multiple guarded specializations
```

当前 exact guard fields：

```text
input count
shape
dtype
device
requires_grad
grad mode
```

核心 API：

```python
from mini_pytorch.compiler import compile_graph

compiled = compile_graph(fn)
out = compiled(x)
print(compiled.stats())
print(compiled.explain(x).format())
```

当前验证基线：

```text
192 / 192 tests passed
```

## 4. 编译学习链

```text
v0.11
Graph IR
   ↓
v0.12
Guards + Graph Cache + Recompile
   ↓
v0.13
AOTAutograd-like Graph Differentiation
   ↓
v0.14
Lowering / Fusion
   ↓
v0.15
Runtime / Performance
   ↓
v1.0
Integrated Architecture
```

## 5. 推荐阅读 v0.12

```text
mini_pytorch/compiler/ir.py
      ↓
mini_pytorch/compiler/capture.py
      ↓
mini_pytorch/compiler/signature.py
      ↓
mini_pytorch/compiler/guards.py
      ↓
mini_pytorch/compiler/cache.py
      ↓
mini_pytorch/compiler/compiled.py
      ↓
tests/test_graph_ir.py
      ↓
tests/test_graph_cache.py
      ↓
docs/GRAPH_CACHE.md
      ↓
examples/15_graph_cache.py
```

## 6. Quick run

```bash
python examples/15_graph_cache.py
python -m unittest discover -s tests -v
```

## 7. 下一版本

`v0.13` 将回答：

> 当 forward 已经是显式 Graph 后，backward 是否也可以从 eager AutogradEngine 中“抽出来”，成为可以检查、变换和执行的 graph program？

目标不是完整复制 PyTorch AOTAutograd，而是先建立：

```text
forward graph
   +
saved forward values
   ↓
backward graph
   ↓
grad outputs → input grads
```
