# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行的 Python/NumPy 代码，一层一层拆开 PyTorch 的核心机制。

**Latest integrated version:** `v0.5.0`

**Current main milestone:** Operator Registry + Dispatcher + Redispatch

mini-pytorch 的目标不是复制完整 PyTorch，而是按照架构依赖顺序回答：

```text
Tensor 是什么？
      ↓
Autograd 为什么能自动 backward？
      ↓
Module / Parameter / Optimizer 如何组织训练状态？
      ↓
数据如何经过 Dataset / Sampler / DataLoader？
      ↓
Storage / stride / view 为什么重要？
      ↓
一个 operator 为什么需要 Dispatcher？
      ↓
如何继续扩展 CNN / device / distributed / compiler？
```

## 1. 版本学习地图

| Version | 核心主题 | 最重要的问题 |
|---|---|---|
| `v0.1` | Tensor + Dynamic Autograd | `backward()` 到底如何自动求导？ |
| `v0.2` | Optimizer + LR Scheduler | 优化状态为什么独立于模型？ |
| `v0.3` | Dataset + Sampler + DataLoader + MNIST | 数据为什么要拆成 dataset / sampling / batching？ |
| `v0.4` | Storage + stride + views + version counter | Tensor 为什么不等于 ndarray？ |
| `v0.5` | Operator Schema + Dispatcher + Redispatch | 一个 `x + y` 为什么需要路由层？ |
| `v0.6` | CNN operator stack | 如何让 Conv2d 等复杂算子建立在 Dispatcher 上？ |

每个历史分支都有自己的：

```text
README.md   → 学习入口
version.md  → 精确版本边界
Agents.md   → 该分支开发约束
```

## 2. 当前 main 的完整架构

```text
                         Tensor API
                             │
                             ▼
                    Storage-backed Tensor
              shape / stride / storage_offset
                             │
                             ▼
                       logical operator
                             │
                             ▼
                     Operator Schema
                             │
                             ▼
                        Dispatcher
                             │
                    DispatchKeySet
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
             Autograd      NoGrad       CPU
                 │           │           │
                 └──── redispatch ───────┘
                             │
                             ▼
                        NumPy kernel
                             │
                             ▼
                     Tensor / GradFn

Module / Parameter ──► nn ──► Loss
                         │
                         ▼
                      Optimizer
                         │
                      Scheduler

Dataset → Sampler → BatchSampler → DataLoader → Tensor batches
```

## 3. v0.5 当前重点

当前首批经过 Dispatcher 的 operator：

```text
aten::add.Tensor
aten::mul.Tensor
aten::matmul.default
aten::relu.default
aten::sum.default
```

典型 requires-grad 调用：

```text
x * y
  ↓
Tensor.__mul__
  ↓
aten::mul.Tensor
  ↓
DispatchKeySet{Autograd, CPU}
  ↓
Autograd wrapper
  ↓ redispatch(remove Autograd)
CPU kernel
  ↓
attach MulBackward
```

因此 CPU kernel 不再需要知道 dynamic graph policy。

## 4. 从零开始的推荐学习顺序

### 第一阶段 — v0.1：Autograd

```text
mini_pytorch/autograd.py
mini_pytorch/tensor.py
mini_pytorch/ops.py
mini_pytorch/nn/module.py
mini_pytorch/optim/optimizer.py
```

理解：

```text
Function.apply
Context
GradFn
DAG backward
broadcast gradient reduction
Parameter registration
```

### 第二阶段 — v0.2：Optimization

```text
mini_pytorch/optim/optimizer.py
mini_pytorch/optim/sgd.py
mini_pytorch/optim/adam.py
mini_pytorch/optim/lr_scheduler.py
```

理解 parameter groups、optimizer state、momentum、Adam moments 和 scheduler lifecycle。

### 第三阶段 — v0.3：Data pipeline

```text
mini_pytorch/data/dataset.py
mini_pytorch/data/sampler.py
mini_pytorch/data/dataloader.py
mini_pytorch/data/mnist.py
examples/06_mnist_mlp.py
```

理解 Dataset / Sampler / BatchSampler / collate 的职责分离。

### 第四阶段 — v0.4：Tensor memory model

```text
mini_pytorch/storage.py
mini_pytorch/tensor.py
tests/test_storage.py
docs/TENSOR_MEMORY.md
examples/07_tensor_memory.py
```

理解 Storage、stride、offset、view、reshape、contiguous、alias 和 inplace version safety。

### 第五阶段 — v0.5：Dispatcher

```text
mini_pytorch/dispatcher.py
        ↓
mini_pytorch/kernels.py
        ↓
mini_pytorch/operator_registry.py
        ↓
mini_pytorch/tensor.py
        ↓
mini_pytorch/ops.py
        ↓
tests/test_dispatcher.py
        ↓
docs/DISPATCHER.md
        ↓
examples/08_dispatcher.py
```

重点理解：

```text
schema != kernel
DispatchKeySet
priority
fallback
redispatch
Autograd wrapper
backend kernel
```

## 5. 快速运行

要求 Python 3.10+。

```bash
pip install -e .
python -m unittest discover -s tests -v
```

运行当前 Dispatcher 教学示例：

```bash
python examples/08_dispatcher.py
```

运行 Tensor memory 示例：

```bash
python examples/07_tensor_memory.py
```

查看 MNIST 训练入口：

```bash
python examples/06_mnist_mlp.py --help
```

## 6. 当前测试基线

v0.5 release candidate 和 frozen branch 均已经通过 GitHub Actions 全量回归：

```text
80 tests
OK
```

覆盖：

- dynamic Autograd
- classification
- Dataset/DataLoader/MNIST
- optimizers
- LR schedulers
- Storage / stride / views
- inplace version safety
- operator schema / registry
- CPU / Autograd / NoGrad dispatch
- redispatch and error paths

## 7. 当前刻意没有实现

为了保持“一版一个核心 mental model”，main 当前还没有：

- Conv2d / MaxPool2d
- BatchNorm / Dropout 完整 operator stack
- real CUDA/device backend
- `.to(device)`
- multiprocessing DataLoader
- advanced Autograd graph lifetime
- DDP
- Dynamo-like graph capture
- AOTAutograd-like lowering
- Inductor-like kernel fusion

这些内容按照 [`Plan.md`](./Plan.md) 逐版本推进。

## 8. 下一版本：v0.6

下一步不应该直接跳到 CUDA，而是先利用刚完成的 Dispatcher 承载更复杂的神经网络 operator：

```text
Conv2d reference kernel
       ↓
Dispatcher registration
       ↓
Autograd wrapper
       ↓
Conv2d backward
       ↓
MaxPool2d
Dropout
BatchNorm
       ↓
CNN MNIST
```

这样可以验证 v0.5 的 Operator/Dispatcher 架构是否真的能承载非平凡算子，而不只是 `add/mul` demo。

## 9. 项目开发规则

继续开发前先阅读：

```text
Agents.md
Plan.md
version.md
```

核心原则始终是：

> **main 保持最新；版本分支保持可学习；每个版本只引入一个主要架构 mental model。**
