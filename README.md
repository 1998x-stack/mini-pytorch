# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行的 Python/NumPy 代码，一层一层拆开 PyTorch 的核心机制。

**Latest integrated version:** `v0.6.0`

**Current main milestone:** CNN Operator Stack built on Dispatcher

**Next milestone:** `v0.7` Device / Backend abstraction

mini-pytorch 不追求复制完整 PyTorch API，而是按照架构依赖顺序回答：

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
复杂 CNN operator 如何建立在 Dispatcher 上？
      ↓
如何进一步支持 device / backend / distributed / compiler？
```

## 1. 版本学习地图

| Version | 核心主题 | 最重要的问题 |
|---|---|---|
| `v0.1` | Tensor + Dynamic Autograd | `backward()` 到底如何自动求导？ |
| `v0.2` | Optimizer + LR Scheduler | 优化状态为什么独立于模型？ |
| `v0.3` | Dataset + Sampler + DataLoader + MNIST | 数据为什么拆成 dataset / sampling / batching？ |
| `v0.4` | Storage + stride + views + version counter | Tensor 为什么不等于 ndarray？ |
| `v0.5` | Operator Schema + Dispatcher + Redispatch | 一个 `x + y` 为什么需要路由层？ |
| `v0.6` | Conv2d / Pooling / Dropout / BatchNorm / CNN | 复杂 NN operator 如何建立在 Dispatcher 上？ |
| `v0.7` | Device / Backend | 多设备为什么不能靠散落的 `if device`？ |

每个历史 branch 都有：

```text
README.md   → 学习入口
version.md  → 精确版本边界
Agents.md   → 该分支开发约束
```

## 2. 当前 main 架构

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
                       backend kernel
                             │
                             ▼
                       Tensor / GradFn

                         nn.Module
                             │
                             ▼
                     functional NN op
                             │
                             └──────────► Dispatcher

Dataset → Sampler → BatchSampler → DataLoader → Tensor batches

Module / Parameter → Optimizer → LR Scheduler
```

## 3. v0.6 最新能力

### Dispatcher-backed Conv2d

```text
nn.Conv2d
   ↓
functional.conv2d
   ↓
aten::conv2d.default
   ↓
Dispatcher
   ↓
Autograd wrapper
   ↓ redispatch
CPU Conv2d kernel
```

提供两条 CPU 教学路径：

```text
reference nested loops
          +
im2col + matrix multiplication
```

两条路径必须数值等价。

### CNN operator stack

当前包含：

- `nn.Conv2d`
- `nn.MaxPool2d`
- `nn.Dropout`
- `nn.BatchNorm1d`
- `nn.BatchNorm2d`
- functional `conv2d / max_pool2d / dropout / batch_norm`
- Conv2d input / weight / bias backward
- BatchNorm Parameter / Buffer state semantics

### CNN MNIST

```text
MNIST
  ↓
DataLoader
  ↓
Conv2d → BatchNorm → ReLU → Pool
  ↓
Conv2d → ReLU → Pool
  ↓
Flatten → Dropout → Linear
  ↓
CrossEntropy
  ↓
AdamW + Scheduler
```

完整示例：

```text
examples/09_cnn_mnist.py
```

并保留：

- Loguru 日志
- metrics.csv
- loss_curve.png
- accuracy_curve.png
- best / last checkpoint

## 4. 推荐学习顺序

### v0.1 — Autograd

```text
mini_pytorch/autograd.py
mini_pytorch/tensor.py
mini_pytorch/ops.py
mini_pytorch/nn/module.py
```

### v0.2 — Optimization

```text
mini_pytorch/optim/optimizer.py
mini_pytorch/optim/sgd.py
mini_pytorch/optim/adam.py
mini_pytorch/optim/lr_scheduler.py
```

### v0.3 — Data pipeline

```text
mini_pytorch/data/dataset.py
mini_pytorch/data/sampler.py
mini_pytorch/data/dataloader.py
mini_pytorch/data/mnist.py
examples/06_mnist_mlp.py
```

### v0.4 — Tensor memory model

```text
mini_pytorch/storage.py
mini_pytorch/tensor.py
docs/TENSOR_MEMORY.md
examples/07_tensor_memory.py
```

### v0.5 — Dispatcher

```text
mini_pytorch/dispatcher.py
mini_pytorch/operator_registry.py
mini_pytorch/kernels.py
docs/DISPATCHER.md
examples/08_dispatcher.py
```

### v0.6 — CNN operator stack

```text
mini_pytorch/kernels.py
        ↓
mini_pytorch/nn_ops.py
        ↓
mini_pytorch/operator_registry.py
        ↓
mini_pytorch/functional.py
        ↓
mini_pytorch/nn/conv.py
mini_pytorch/nn/pooling.py
mini_pytorch/nn/dropout.py
mini_pytorch/nn/batchnorm.py
        ↓
tests/test_cnn_ops.py
        ↓
docs/CNN_OPERATORS.md
        ↓
examples/09_cnn_mnist.py
```

## 5. 快速运行

Python 3.10+：

```bash
pip install -e .
python -m unittest discover -s tests -v
```

当前集成测试基线：

```text
90 tests
OK
```

运行 Dispatcher 示例：

```bash
python examples/08_dispatcher.py
```

运行 CNN MNIST 教学规模：

```bash
python examples/09_cnn_mnist.py \
  --epochs 2 \
  --train-limit 1000 \
  --test-limit 300 \
  --implementation im2col
```

## 6. 当前明确限制

截至 v0.6，项目还没有：

- real CUDA;
- `.to(device)`;
- backend-owned allocation;
- grouped/depthwise Conv2d；
- distributed training；
- graph capture；
- compiler lowering/fusion。

这些由后续版本按依赖顺序引入。

## 7. 为什么 v0.7 是 Device / Backend

v0.5 已经有 Dispatcher，v0.6 已证明 Dispatcher 能承载非平凡 CNN operator。

下一步自然是：

```text
Tensor / Storage
      ↓
Device metadata
      ↓
Dispatcher
   ┌────┴────┐
   ▼         ▼
 CPU       accelerator backend
```

v0.7 要解决的不是“多写一个 CUDA if”，而是：

> allocation、Tensor metadata、operator routing 和 backend implementation 应如何组合，才能让同一个 Tensor/operator API 支持不同设备。

详细版本计划见 `Plan.md`。