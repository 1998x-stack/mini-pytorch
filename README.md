# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行的 Python/NumPy 代码，一层一层拆开 PyTorch 的核心机制。

**Latest integrated version:** `v0.7.0`

**Current main milestone:** Device / Backend abstraction

**Next milestone:** `v0.8` Advanced Autograd Engine

mini-pytorch 不追求复制完整 PyTorch API，而是按照架构依赖顺序回答：

```text
Tensor 是什么？
      ↓
Autograd 为什么能 backward？
      ↓
Module / Parameter / Optimizer 如何组织训练状态？
      ↓
Dataset / Sampler / DataLoader 为什么拆开？
      ↓
Storage / stride / view 为什么重要？
      ↓
operator 为什么需要 Dispatcher？
      ↓
Conv2d 等复杂 NN operator 如何建立在 Dispatcher 上？
      ↓
Tensor 到底住在哪里，backend 如何拥有 allocation？
      ↓
Autograd engine 如何管理 graph lifetime / hooks / repeated backward？
```

## 1. 版本学习地图

| Version | 核心主题 | 最重要的问题 |
|---|---|---|
| `v0.1` | Tensor + Dynamic Autograd | `backward()` 如何自动求导？ |
| `v0.2` | Optimizer + LR Scheduler | 优化状态为什么独立于模型？ |
| `v0.3` | Dataset + Sampler + DataLoader + MNIST | 数据为什么拆成 storage / sampling / batching？ |
| `v0.4` | Storage + stride + views + version counter | Tensor 为什么不等于 ndarray？ |
| `v0.5` | Operator Schema + Dispatcher + Redispatch | 一个 `x + y` 为什么需要路由层？ |
| `v0.6` | Conv2d / Pooling / Dropout / BatchNorm | 复杂 NN operator 如何复用 Dispatcher？ |
| `v0.7` | Device / Backend | 多 backend 为什么不能靠散落的 `if device`？ |
| `v0.8` | Advanced Autograd Engine | graph lifetime / hooks / retain_graph 如何工作？ |

每个历史版本 branch 都有：

```text
README.md   → 本版本学习入口
version.md  → 与上一版的精确 delta
Agents.md   → 本版本工程约束
```

## 2. 当前 main 完整架构

```text
                             Tensor
                shape / stride / dtype / device
                               │
                               ▼
                             Storage
              backend-owned allocation + version
                               │
                               ▼
                        logical operator
                               │
                               ▼
                     Operator Schema
                               │
                               ▼
                           Dispatcher
                  policy keys + backend keys
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
          Autograd           NoGrad          CPU / EDU
             │                                   │
             └────────── redispatch ─────────────┘
                               │
                               ▼
                         numerical kernel
                               │
                               ▼
                         Tensor / GradFn

nn.Module → functional NN op ─────────────────► Dispatcher

Dataset → Sampler → BatchSampler → DataLoader → Tensor batches

Module / Parameter → Optimizer → LR Scheduler
```

`EDU` 是 v0.7 的模拟教育 backend：拥有独立 Device / Backend / DispatchKey，但底层仍使用 NumPy host memory，从而可以在普通 CI 上验证多 backend 架构。它不是 CUDA，也不代表 GPU 加速。

## 3. v0.7 最新能力

### Device 是一等值对象

```python
from mini_pytorch import Device, Tensor

cpu = Device("cpu")
edu = Device("edu")

x = Tensor([1, 2, 3], device=edu)
print(x.device)
# edu
```

### Storage 由 backend 拥有

```text
Storage
├── Device
├── Backend
├── backend-owned allocation
└── VersionCounter
```

当前 backend registry：

```text
cpu → NumPyBackend
edu → EducationalBackend
```

### Tensor.to(device)

```python
x = Tensor([1.0, 2.0], requires_grad=True)
y = x.to("edu")
```

语义：

```text
same device + copy=False → return self
same device + copy=True  → new Storage
cross-device             → new target Storage
```

跨设备 transfer 按逻辑值复制，因此断开 Storage alias；但在 grad mode 下通过 CopyBackward 保持梯度链路。

### Device-aware Dispatcher

```text
CPU + grad  → {Autograd, CPU}
EDU + grad  → {Autograd, EDU}
CPU no_grad → {NoGrad, CPU}
EDU no_grad → {NoGrad, EDU}
```

因此 Autograd 仍是 policy layer，backend identity 在 redispatch 之后继续存在。

### Mixed-device protection

```python
Tensor([1.0], device="cpu") + Tensor([2.0], device="edu")
```

会在 backend math 前明确报 device mismatch，而不是偷偷复制。

### Module.to(device)

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, 3),
    nn.BatchNorm2d(8),
)

model.to("edu")
```

会递归迁移：

```text
Parameters
Buffers
child Modules
```

同时保持 Parameter 对象和注册关系。

## 4. 从零阅读源码的推荐顺序

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
mini_pytorch/nn_ops.py
mini_pytorch/operator_registry.py
mini_pytorch/functional.py
mini_pytorch/nn/conv.py
mini_pytorch/nn/pooling.py
mini_pytorch/nn/dropout.py
mini_pytorch/nn/batchnorm.py
docs/CNN_OPERATORS.md
examples/09_cnn_mnist.py
```

### v0.7 — Device / Backend

```text
mini_pytorch/device.py
        ↓
mini_pytorch/backend.py
        ↓
mini_pytorch/storage.py
        ↓
mini_pytorch/tensor.py
        ↓
mini_pytorch/dispatcher.py
        ↓
mini_pytorch/operator_registry.py
        ↓
mini_pytorch/nn/module.py
        ↓
tests/test_device_backend.py
        ↓
docs/DEVICE_BACKEND.md
        ↓
examples/10_device_backend.py
```

## 5. 快速运行

Python 3.10+：

```bash
pip install -e .
python -m unittest discover -s tests -v
```

当前集成基线：

```text
103 tests
OK
```

运行 Device / Backend 示例：

```bash
python examples/10_device_backend.py
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

截至 v0.7，项目刻意还没有：

- real CUDA / CuPy；
- async device copy；
- pinned memory；
- streams/events；
- mixed precision；
- caching allocator；
- grouped/depthwise Conv2d beyond current boundary；
- distributed training；
- graph capture；
- compiler lowering/fusion。

这些由后续版本按依赖顺序引入。

## 7. 为什么下一版是 Advanced Autograd

现在框架已经具备：

```text
Tensor memory identity
        +
operator routing
        +
NN operators
        +
device/backend placement
```

但当前 Autograd engine 仍然非常朴素：graph 构建后长期留在对象上，non-leaf grad 不保留，没有 hooks，也没有 `retain_graph` 的 graph lifecycle。

因此 v0.8 应开始回答：

```text
forward graph
    ↓
backward traversal
    ↓
什么时候释放 saved state？
    ↓
为什么第二次 backward 会失败？
    ↓
retain_graph=True 保留了什么？
    ↓
non-leaf retain_grad() 怎么实现？
    ↓
gradient hook 在什么时机执行？
```

详细任务见 `Plan.md`。
