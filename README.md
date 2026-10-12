# mini-pytorch

一个为了**快速理解 PyTorch 核心机制**而设计的可运行教学框架。

目标不是复刻完整 PyTorch，而是用约千行可读 Python/NumPy，把下面这条主链真正跑通：

```text
Tensor
  ↓
Function / Operator
  ↓
Dynamic Computation Graph
  ↓
Autograd / backward
  ↓
Parameter
  ↓
Module / Linear / Sequential
  ↓
Loss
  ↓
Optimizer
  ↓
SGD / Adam
  ↓
Training Loop
```

项目刻意保持了和 PyTorch 相似的抽象边界，因此可以继续扩展更多算子、优化器、层、设备后端和序列化能力。

---

## 1. 你会真正学懂什么

### Tensor 不只是 ndarray

`Tensor` = 数据 + 梯度需求 + 动态计算图入口。

```python
x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)
z = x * y + x ** 2
z.backward()
```

这里会经历：

1. `Mul.apply` / `Pow.apply` / `Add.apply` 执行 forward；
2. 每个输出 Tensor 保存自己的 `GradFn`；
3. `backward()` 从输出开始做拓扑排序；
4. 每个算子的 backward 根据链式法则回传梯度；
5. 同一个 Tensor 被多条路径使用时，梯度自动累加。

### grad_fn 是什么

本项目中：

```python
print(z.grad_fn)
# AddBackward
```

它对应真实 PyTorch 中类似：

```text
<AddBackward0>
```

`GradFn` 保存：

- 当前算子的 Function 类型；
- 当前节点的父 Tensor；
- forward 为 backward 保存的 Context。

### 为什么广播反向传播要 sum

例如：

```python
x.shape == (32, 128)
b.shape == (128,)
y = x + b
```

forward 时 `b` 被广播成 32 行。

backward 时 `db` 必须把 batch 维的梯度重新相加。

本项目把这件事集中在：

```text
mini_pytorch/ops.py -> unbroadcast()
```

这是理解 PyTorch broadcasting + autograd 的一个非常重要的细节。

---

## 2. 目录结构

```text
mini-pytorch/
├── mini_pytorch/
│   ├── tensor.py            # Tensor、backward、算子语法
│   ├── autograd.py          # Function / Context / GradFn / no_grad
│   ├── ops.py               # 可微分算子
│   ├── functional.py        # 类似 torch.nn.functional
│   ├── graph.py             # 导出计算图 DOT
│   ├── utils.py             # gradcheck
│   ├── nn/
│   │   ├── module.py        # Module / Parameter / 参数注册
│   │   ├── linear.py        # Linear
│   │   ├── activation.py    # ReLU / Sigmoid
│   │   ├── loss.py          # MSELoss
│   │   └── container.py     # Sequential
│   └── optim/
│       ├── optimizer.py     # Optimizer 抽象层 + parameter groups
│       ├── sgd.py           # SGD + momentum + weight decay
│       └── adam.py          # Adam
├── examples/
│   ├── 01_autograd.py
│   ├── 02_linear_regression.py
│   ├── 03_mlp_xor.py
│   └── 04_parameter_groups.py
├── tests/
│   ├── test_autograd.py
│   ├── test_ops.py
│   ├── test_nn.py
│   └── test_optim.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXTENDING.md
│   └── LEARNING_PATH.md
├── pyproject.toml
└── README.md
```

---

## 3. 快速运行

要求：Python 3.10+。

安装：

```bash
pip install -e .
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

运行最小 autograd 示例：

```bash
python examples/01_autograd.py
```

训练线性回归：

```bash
python examples/02_linear_regression.py
```

训练 XOR：

```bash
python examples/03_mlp_xor.py
```

---

## 4. 和真实 PyTorch 的一一映射

| mini-pytorch | PyTorch | 核心意义 |
|---|---|---|
| `Tensor` | `torch.Tensor` | 数据与 autograd 元数据 |
| `GradFn` | `grad_fn` | 动态计算图节点 |
| `Function` | `torch.autograd.Function` | 定义 forward/backward |
| `Context` | `ctx` | forward 保存 backward 所需信息 |
| `Tensor.backward()` | `Tensor.backward()` | 反向模式自动微分 |
| `Parameter` | `nn.Parameter` | 可训练参数 |
| `Module` | `nn.Module` | 参数/子模块注册与组织 |
| `Linear` | `nn.Linear` | 仿射层 |
| `Sequential` | `nn.Sequential` | 层级组合 |
| `MSELoss` | `nn.MSELoss` | 损失函数 |
| `Optimizer` | `optim.Optimizer` | 优化器抽象层 |
| `SGD` | `optim.SGD` | 一阶优化 |
| `Adam` | `optim.Adam` | 自适应一阶优化 |
| `no_grad` | `torch.no_grad` | 禁止构图 |
| `gradcheck` | `torch.autograd.gradcheck` | 数值梯度验证 |

---

## 5. 为什么不是把所有代码写在一个 Tensor 类里

很多教学版 autograd 项目会把：

- forward；
- backward；
- graph；
- optimizer；
- neural network；

全部塞到一个文件。

这样适合理解链式法则，但不适合理解 PyTorch 的软件架构。

本项目故意分层：

```text
Tensor API
   ↓
Function / Op
   ↓
Autograd graph

Module API
   ↓
Parameter registration

Optimizer API
   ↓
SGD / Adam / future optimizers
```

这样以后新增功能时只扩展相应层。

---

## 6. 如何新增一个算子

例如实现 `tanh`。

第一步，在 `ops.py` 中定义：

```python
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.tanh(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return (grad_output * (1 - out * out),)
```

第二步，在 Tensor API 中暴露：

```python
def tanh(self):
    from .ops import Tanh
    return Tanh.apply(self)
```

不需要修改：

- backward engine；
- Module；
- Optimizer；
- SGD；
- Adam。

这就是扩展边界。

---

## 7. 如何新增一个优化器

所有优化器继承：

```python
class Optimizer:
    ...
```

新增 RMSprop 时，只需要：

```python
class RMSprop(Optimizer):
    def __init__(self, params, ...):
        super().__init__(params, defaults={...})

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                ...
```

`Optimizer` 已统一提供：

- parameter groups；
- optimizer state；
- `zero_grad()`；
- `state_dict()`；
- 重复参数检查。

因此模型代码完全不用改。

---

## 8. Adam 为什么能自然接入

训练代码：

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

替换为：

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

训练循环完全不变：

```python
pred = model(x)
loss = criterion(pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这正是 Optimizer abstraction 的价值。

---

## 9. Parameter Groups

本项目支持类似 PyTorch 的 parameter groups：

```python
optimizer = optim.SGD(
    [
        {"params": model[0].parameters(), "lr": 1e-2},
        {"params": model[2].parameters(), "lr": 1e-3},
    ],
    lr=1e-2,
)
```

因此以后可以继续增加：

- 不同层不同学习率；
- 不同 weight decay；
- layer-wise learning-rate decay；
- fine-tuning；
- frozen / unfrozen parameter groups。

---

## 10. Module 自动注册机制

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
```

为什么：

```python
list(model.parameters())
```

能自动找到两层参数？

因为 `Module.__setattr__` 会识别：

```text
Parameter -> _parameters
Module    -> _modules
Tensor buffer -> _buffers
```

这也是理解真实 `nn.Module` 最关键的机制之一。

---

## 11. state_dict

```python
state = model.state_dict()
```

返回类似：

```text
fc1.weight
fc1.bias
fc2.weight
fc2.bias
```

它不是“整个 Python 对象的序列化”，而是模型状态的**名字到数组的映射**。

这正是 PyTorch `state_dict` 思想的简化版。

---

## 12. backward 的核心算法

`Tensor.backward()` 的核心流程：

```text
output
  ↓
DFS 构造拓扑序
  ↓
seed gradient
  ↓
reverse topological traversal
  ↓
GradFn.backward
  ↓
parent gradients
  ↓
gradient accumulation
```

最关键的不是求导公式，而是：

> 一个通用反向模式自动微分引擎，如何遍历任意动态 DAG。

---

## 13. 为什么非标量 Tensor backward 需要 grad

如果：

```python
y.shape == (10,)
```

则不存在唯一的“dy/dx 数字”。

反向模式实际计算的是 vector-Jacobian product：

```text
v^T J
```

因此：

```python
y.backward()
```

会报错，而：

```python
y.backward(np.ones_like(y.data))
```

是有定义的。

这和真实 PyTorch 的行为一致。

---

## 14. 当前 MVP 的刻意限制

为了让源码保持足够小且清晰，目前没有实现：

- CUDA / device；
- dtype promotion；
- stride / storage / view semantics；
- inplace operation / version counter；
- retain_graph；
- higher-order autograd；
- convolution；
- BatchNorm；
- Dropout；
- DataLoader；
- distributed / DDP；
- AMP；
- torch.compile；
- C++ dispatcher。

这些不是“做不到”，而是按学习顺序刻意延后。

---

## 15. 推荐学习顺序

### 第一阶段：只看 Autograd

按顺序阅读：

```text
tensor.py
  ↓
autograd.py
  ↓
ops.py
  ↓
tests/test_autograd.py
```

重点理解：

```text
Function.apply
Context
GradFn
backward topology
gradient accumulation
unbroadcast
```

### 第二阶段：看 nn.Module

```text
nn/module.py
  ↓
nn/linear.py
  ↓
nn/container.py
```

重点理解：

```text
Parameter
__setattr__
_parameters
_modules
state_dict
```

### 第三阶段：看优化器

```text
optim/optimizer.py
  ↓
optim/sgd.py
  ↓
optim/adam.py
```

重点理解：

```text
parameter groups
optimizer state
momentum buffer
first moment
second moment
bias correction
```

### 第四阶段：跑训练

```text
examples/02_linear_regression.py
examples/03_mlp_xor.py
```

最终确保你能够完整解释：

```python
pred = model(x)
loss = criterion(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这五行内部到底发生了什么。

---

## 16. 下一阶段最值得扩展的内容

推荐按这个顺序：

```text
V0.1 当前版本
│
├─ Tensor + dynamic autograd
├─ broadcasting
├─ Module / Parameter
├─ Linear / ReLU / Sigmoid
├─ SGD / Adam
└─ gradcheck

V0.2 Tensor semantics
│
├─ Storage
├─ stride
├─ contiguous
├─ view vs reshape
├─ transpose
└─ inplace + version counter

V0.3 NN ecosystem
│
├─ Softmax
├─ CrossEntropyLoss
├─ Embedding
├─ Conv2d
├─ Dropout
└─ BatchNorm

V0.4 Data
│
├─ Dataset
├─ Sampler
└─ DataLoader

V0.5 Device abstraction
│
├─ Backend protocol
├─ CPU NumPy backend
└─ optional CuPy backend

V0.6 Advanced autograd
│
├─ retain_grad
├─ retain_graph
├─ create_graph
└─ higher-order gradients

V0.7 PyTorch internals mapping
│
├─ Dispatcher
├─ ATen
├─ Autograd Engine
├─ CUDA allocator
└─ torch.compile / Inductor / Triton
```

---

## 17. 设计原则

本项目优先级严格是：

```text
可读性 > 架构清晰 > 正确性 > 可扩展性 > 性能
```

真正的 PyTorch 则需要同时解决：

```text
性能
硬件
并发
分布式
dtype
device
内存
ABI
编译器
生态兼容
```

因此不要把 mini-pytorch 当生产框架；把它当成阅读真实 PyTorch 源码前的一张“结构地图”。
