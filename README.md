# mini-pytorch v0.5 — Operator Registry + Dispatcher

> 从这一版开始，`Tensor.__add__` 不再直接决定“怎么计算”，而是只负责调用一个**逻辑 operator**。

**版本主题：** Operator Schema → Registry → DispatchKeySet → Dispatcher → Redispatch

**状态：** active development / release candidate

## 1. 为什么 PyTorch 需要 Dispatcher

如果没有 Dispatcher，Tensor 很容易变成：

```python
if device == "cpu":
    ...
elif device == "cuda":
    ...
if requires_grad:
    ...
if autocast:
    ...
```

功能越多，Tensor 方法越不可维护。

v0.5 改成：

```text
x + y
  ↓
Tensor.__add__
  ↓
aten::add.Tensor
  ↓
Dispatcher
  ↓
DispatchKeySet
  ↓
selected implementation
```

一个逻辑 operator 可以拥有多个实现。

## 2. 当前 DispatchKey

教学版只保留四个 key：

```text
Autograd
NoGrad
CUDA      ← reserved, v0.5 不实现真实 CUDA
CPU
```

优先级：

```text
Autograd > NoGrad > CUDA > CPU
```

普通 Tensor：

```text
{CPU}
  ↓
CPU kernel
```

`requires_grad=True`：

```text
{Autograd, CPU}
      ↓
Autograd wrapper
      ↓ redispatch(remove Autograd)
{CPU}
      ↓
CPU kernel
```

`no_grad()`：

```text
{NoGrad, CPU}
      ↓
NoGrad fallback
      ↓ redispatch(remove NoGrad)
{CPU}
      ↓
CPU kernel
```

## 3. Schema 和 Kernel 不是一回事

Schema 描述“这个 operator 是什么”：

```text
aten::add.Tensor
(Tensor self, Tensor other) -> Tensor
```

Kernel 描述“某个 dispatch key 下怎么实现”：

```text
aten::add.Tensor
├── Autograd → autograd wrapper
└── CPU      → NumPy add kernel
```

这使 API identity 与 implementation identity 分离。

## 4. v0.5 首批 Dispatcher operator

- `aten::add.Tensor`
- `aten::mul.Tensor`
- `aten::matmul.default`
- `aten::relu.default`
- `aten::sum.default`

对应 Tensor API：

```python
x + y
x * y
x @ y
x.relu()
x.sum()
```

其他算子暂时仍走 v0.4 的直接 `Function.apply()` 路径，便于清晰对比重构前后架构。

## 5. Autograd 为什么是 dispatch layer

以 `x * x` 为例：

```text
Tensor.__mul__
   ↓
Dispatcher
   ↓ choose Autograd
Autograd Mul wrapper
   ├── save backward context
   └── redispatch
          ↓
        CPU Mul kernel
          ↓
        Tensor result
          ↓
      attach MulBackward
```

因此 CPU kernel 只需要知道：

```python
return a * b
```

它不需要知道：

- `requires_grad`
- `GradFn`
- graph construction
- `no_grad()`

这就是 backend kernel 与 Autograd policy 分离的价值。

## 6. 核心源码

推荐阅读：

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

### `dispatcher.py`

包含：

- `OperatorSchema`
- `DispatchKey`
- `DispatchKeySet`
- `Dispatcher`
- `DispatchDecision`
- `Library`
- global `dispatcher`

### `kernels.py`

只包含 NumPy numerical kernels，不包含 Autograd。

### `operator_registry.py`

把 logical operator 与：

```text
CPU implementation
Autograd implementation
NoGrad fallback
```

连接起来。

## 7. 快速运行

```bash
pip install -e .
python -m unittest discover -s tests -v
python examples/08_dispatcher.py
```

## 8. 一个最小 introspection 示例

```python
from mini_pytorch import Tensor, dispatcher

x = Tensor([1.0, 2.0], requires_grad=True)
y = Tensor([3.0, 4.0])

print(dispatcher.explain("aten::add.Tensor", x, y))
```

应该看到类似：

```text
selected_key = Autograd
keyset       = {Autograd, CPU}
```

而在：

```python
with no_grad():
    ...
```

中会变成：

```text
selected_key = NoGrad
implementation = fallback
```

## 9. v0.5 从 v0.4 继承什么

以下机制保持稳定：

- Storage
- shape / stride / storage_offset
- view / reshape / contiguous
- alias semantics
- version counter
- inplace safety
- dynamic backward engine
- Module / Optimizer / Scheduler
- Dataset / Sampler / DataLoader
- MNIST classification stack

Dispatcher 是插入在 Tensor API 与 operator implementation 之间的新层。

## 10. 本版本不做什么

- 不实现真实 CUDA
- 不实现 `.to(device)`
- 不实现完整 PyTorch DispatchKey taxonomy
- 不实现 Tensor subclass dispatch
- 不实现 functionalization
- 不实现 Conv2d
- 不实现 DDP
- 不实现 compiler

## 11. 下一版本

`v0.6` 将利用 Dispatcher 承载更复杂的神经网络算子：

```text
Conv2d
MaxPool2d
Dropout
BatchNorm
CNN MNIST
```

而不是重新把这些逻辑硬编码回 Tensor。

详细版本边界见 [`version.md`](./version.md)，Dispatcher 深度讲解见 [`docs/DISPATCHER.md`](./docs/DISPATCHER.md)。