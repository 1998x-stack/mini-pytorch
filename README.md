# mini-pytorch v0.6 — CNN Operator Stack

> 用 Conv2d / Pooling / Dropout / BatchNorm 把 v0.5 的 Dispatcher 真正带进神经网络算子层。

**版本主题：** Module → Functional API → Dispatcher → Autograd wrapper → CPU kernel

**Package version：** `0.6.0`

**当前验证基线：** `90/90 tests passed`（核心实现 release candidate）

## 1. v0.6 相比 v0.5 增加了什么

v0.5 解决：

```text
一个 logical operator
为什么需要 schema / registry / DispatchKey / redispatch？
```

v0.6 继续解决：

```text
复杂神经网络 operator
如何建立在 Dispatcher 上？
```

核心链路：

```text
nn.Conv2d
    ↓
functional.conv2d
    ↓
aten::conv2d.default
    ↓
Dispatcher
    ↓
DispatchKeySet {Autograd, CPU}
    ↓
Autograd wrapper
    ↓ redispatch(remove Autograd)
CPU Conv2d kernel
    ↓
Tensor + Conv2dOpBackward
```

CPU kernel 仍然只负责数值计算，不知道 Module、Optimizer 或动态图策略。

## 2. 本版本新增能力

### Conv2d

```python
from mini_pytorch import Tensor, nn

layer = nn.Conv2d(
    1,
    8,
    kernel_size=3,
    padding=1,
    implementation="im2col",
)

x = Tensor(...)
y = layer(x)
```

支持：

- NCHW input
- OIHW weight
- bias
- stride
- padding
- dilation
- `groups=1`
- input / weight / bias backward
- readable nested-loop reference kernel
- im2col + matrix multiplication kernel

`groups != 1` 会显式报错，而不是假装支持。

### MaxPool2d

```python
pool = nn.MaxPool2d(2, stride=2)
y = pool(x)
```

backward 只把上游梯度传回每个 pooling window 的最大值位置。

### Dropout

```python
dropout = nn.Dropout(0.2, seed=42)

model.train()
y = dropout(x)

model.eval()
y = dropout(x)  # identity
```

训练使用 inverted dropout：

```text
mask / (1-p)
```

因此 eval 不需要额外缩放。

### BatchNorm1d / BatchNorm2d

```python
bn = nn.BatchNorm2d(8)
```

明确区分：

```text
weight / bias
    ↓
Parameter
    ↓
Optimizer 会更新

running_mean / running_var / num_batches_tracked
    ↓
Buffer
    ↓
进入 state_dict，但不进入 parameters()
```

这是理解真实 `nn.Module` 状态管理非常重要的一层。

## 3. Conv2d 为什么保留两套 kernel

### reference

```text
N
↓
C_out
↓
H_out
↓
W_out
↓
C_in
↓
K_h
↓
K_w
```

优点：每一个索引关系都看得见。

### im2col

```text
sliding windows
      ↓
    im2col
      ↓
(N*H_out*W_out, C_in*K_h*K_w)
      ↓
matrix multiplication
      ↓
reshape / transpose
      ↓
NCHW output
```

测试要求：

```text
reference output == im2col output
```

因此优化不会替代教学基线。

## 4. 新增核心文件

```text
mini_pytorch/
├── kernels.py                 # Conv/Pool CPU numerical kernels
├── nn_ops.py                  # Conv2d/MaxPool2d Autograd definitions
├── operator_registry.py       # Dispatcher registration
├── functional.py              # conv2d/max_pool2d/dropout/batch_norm
└── nn/
    ├── conv.py                # nn.Conv2d
    ├── pooling.py             # nn.MaxPool2d
    ├── dropout.py             # nn.Dropout
    └── batchnorm.py           # BatchNorm1d / BatchNorm2d

tests/
└── test_cnn_ops.py

docs/
└── CNN_OPERATORS.md

examples/
└── 09_cnn_mnist.py
```

## 5. 推荐阅读顺序

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

建议重点追踪一次：

```python
out = nn.Conv2d(...)(x)
loss = out.sum()
loss.backward()
```

然后回答：

1. 参数在哪里？
2. logical operator 在哪里？
3. Dispatcher 如何决定 Autograd？
4. Autograd wrapper 保存了什么？
5. 为什么要 redispatch？
6. CPU kernel 为什么不需要知道 `requires_grad`？
7. backward 如何得到 dInput / dWeight / dBias？

## 6. 测试

```bash
pip install -e .
python -m unittest discover -s tests -v
```

当前核心实现 CI：

```text
Ran 90 tests
OK
```

新增覆盖：

- Conv2d known-value forward
- reference vs im2col equivalence
- finite-difference input gradient
- finite-difference weight gradient
- finite-difference bias gradient
- Conv2d Dispatcher Autograd routing
- unsupported groups error
- MaxPool2d forward/backward
- Dropout train/eval/seed
- BatchNorm Parameter vs Buffer
- BatchNorm train/eval running stats
- CNN end-to-end optimizer step

## 7. MNIST CNN

查看参数：

```bash
python examples/09_cnn_mnist.py --help
```

推荐先跑教学规模：

```bash
python examples/09_cnn_mnist.py \
  --epochs 2 \
  --train-limit 1000 \
  --test-limit 300 \
  --implementation im2col
```

输出：

```text
runs/mnist-cnn/<timestamp>/
├── config.json
├── train.log
├── metrics.csv
├── loss_curve.png
├── accuracy_curve.png
├── best_model.npz
└── last_model.npz
```

完整 60,000 / 10,000 MNIST 也可以通过：

```bash
--train-limit 0 --test-limit 0
```

但本项目 kernel 是教学 NumPy 实现，不应把它当成性能版 PyTorch。

## 8. CNN 示例结构

```text
N×1×28×28
   ↓
Conv2d(1 → 8, 3×3)
   ↓
BatchNorm2d(8)
   ↓
ReLU
   ↓
MaxPool2d(2)
   ↓
Conv2d(8 → 16, 3×3)
   ↓
ReLU
   ↓
MaxPool2d(2)
   ↓
Flatten
   ↓
Dropout
   ↓
Linear(16*7*7 → 10)
   ↓
CrossEntropyLoss
```

训练继续复用旧版本能力：

```text
Dataset / DataLoader     ← v0.3
AdamW / Scheduler        ← v0.2
Storage / version safety ← v0.4
Dispatcher               ← v0.5
CNN operators            ← v0.6
```

## 9. 本版本刻意不做什么

v0.6 不实现：

- CUDA
- `.to(device)`
- grouped/depthwise Conv2d
- ConvTranspose
- adaptive pooling
- mixed precision
- production kernel autotuning
- distributed training
- graph compiler

这些不是遗漏，而是版本边界。

## 10. 与上一版的核心差异

```text
v0.5
Tensor op
  ↓
Dispatcher
  ↓
simple CPU kernels

          ↓ v0.6 adds

v0.6
nn.Module
  ↓
functional NN operator
  ↓
Dispatcher
  ↓
Autograd wrapper
  ↓
non-trivial CPU kernels
  ↓
CNN training
```

一句话：

> **v0.5 证明 Dispatcher 能路由 operator；v0.6 证明这个架构能承载真正的神经网络算子。**

## 11. 下一版本

`v0.7` 将进入 **Device / Backend abstraction**：

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

目标是解释：为什么真实框架不能靠在每个 Tensor 方法里堆 `if device == ...` 来支持多后端。