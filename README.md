# mini-pytorch v0.1 — Eager Tensor + Dynamic Autograd MVP

> 一个为了从源码理解 PyTorch 核心训练链路而构建的 NumPy 教学框架。

**版本主题：** `Tensor → Operator → Dynamic Graph → backward → Module → Optimizer`

**状态：** frozen historical learning snapshot

v0.1 是整个项目的基线版本。目标不是复刻完整 PyTorch API，而是让下面五行训练代码背后的机制变得可读、可运行、可调试：

```python
pred = model(x)
loss = loss_fn(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 1. 本版本解决什么问题

v0.1 回答一个最基础的问题：

> 如何用很少的 Python/NumPy 代码实现一个真正可训练的 PyTorch-like 框架？

核心链路：

```text
Tensor
  ↓
Function / Operator
  ↓
Dynamic Computation Graph
  ↓
Reverse-mode Autograd
  ↓
Parameter
  ↓
Module / Linear / Sequential
  ↓
Loss
  ↓
Optimizer
  ↓
Training Loop
```

## 2. 核心能力

### Tensor / Autograd

- `Tensor(data, requires_grad=...)`
- `grad` / `grad_fn` / leaf tensor
- dynamic computation graph
- reverse topological backward
- gradient accumulation
- broadcasting-aware backward
- `detach()`
- `no_grad()`
- numerical `gradcheck`

### Operators

- `+ - * / **`
- matrix multiplication
- `sum()` / `mean()`
- reshape / transpose
- ReLU / sigmoid / exp / log

### Neural network

- `Parameter`
- `Module`
- recursive parameter registration
- `Linear`
- `ReLU` / `Sigmoid`
- `Sequential`
- `MSELoss`
- `state_dict()` / `load_state_dict()`

### Optimizer

- extensible `Optimizer`
- parameter groups
- SGD
- momentum
- weight decay
- Adam

## 3. 推荐源码阅读顺序

```text
mini_pytorch/autograd.py
        ↓
mini_pytorch/tensor.py
        ↓
mini_pytorch/ops.py
        ↓
mini_pytorch/nn/module.py
        ↓
mini_pytorch/nn/linear.py
        ↓
mini_pytorch/optim/optimizer.py
        ↓
mini_pytorch/optim/sgd.py
        ↓
mini_pytorch/optim/adam.py
```

重点理解：

1. `Function.apply()` 如何建立动态计算图；
2. `GradFn` 如何连接父 Tensor；
3. `Tensor.backward()` 如何做 DAG 反向遍历；
4. broadcasting 后为什么 backward 需要 `unbroadcast()`；
5. `Module.__setattr__` 为什么能自动注册 Parameter；
6. Optimizer 为什么独立于 Module。

## 4. 快速运行

```bash
pip install -e .
python -m unittest discover -s tests -v
python examples/01_autograd.py
python examples/02_linear_regression.py
python examples/03_mlp_xor.py
```

## 5. 和真实 PyTorch 的概念映射

| mini-pytorch | PyTorch | 作用 |
|---|---|---|
| `Tensor` | `torch.Tensor` | 数据 + Autograd 元数据 |
| `Function` | `torch.autograd.Function` | forward/backward 边界 |
| `GradFn` | `grad_fn` / Autograd Node | 动态图节点 |
| `Parameter` | `nn.Parameter` | 可训练 Tensor |
| `Module` | `nn.Module` | 模型层级与参数管理 |
| `Optimizer` | `optim.Optimizer` | 参数更新与优化状态 |
| `SGD` | `optim.SGD` | 一阶优化 |
| `Adam` | `optim.Adam` | 自适应优化 |

## 6. v0.1 刻意不实现

这一版暂时不引入：

- LR Scheduler
- 丰富 Optimizer family
- Dataset / Sampler / DataLoader
- MNIST classification
- explicit Storage / stride / storage offset
- view / reshape / contiguous 的真实内存语义
- inplace version counter
- Dispatcher
- Conv2d / CUDA / DDP / compiler

这些内容留给后续版本逐层展开。

## 7. 下一版本

`v0.2` 保持 Tensor/Autograd/Module 基本稳定，专门深入 **Optimizer state + parameter groups + learning-rate scheduling**。

详细版本边界见 [`version.md`](./version.md)。