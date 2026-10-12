# Extending mini-pytorch

## 1. 新增 differentiable operator

原则：算子的微分逻辑只放在 `Function` 子类中。

示例：Tanh。

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
        return (grad_output * (1 - out ** 2),)
```

然后给 Tensor 增加薄封装：

```python
def tanh(self):
    return Tanh.apply(self)
```

必须增加 gradcheck。

## 2. 新增 Layer

新层只需要继承 Module：

```python
class MyLayer(Module):
    def __init__(self, ...):
        super().__init__()
        self.weight = Parameter(...)

    def forward(self, x):
        return ...
```

只要 forward 使用已有 Tensor ops，autograd 自动工作。

## 3. 新增 Optimizer

继承 Optimizer，并把默认超参数交给基类：

```python
class RMSprop(Optimizer):
    def __init__(self, params, lr=..., alpha=...):
        super().__init__(params, defaults={"lr": lr, "alpha": alpha})

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state.setdefault(id(p), {})
                ...
```

不要让 Optimizer 依赖具体网络类型。

## 4. 新增 device backend

推荐未来不要直接在 Tensor 里硬编码 CuPy。

可以引入：

```text
Backend
├── array(...)
├── matmul(...)
├── exp(...)
├── zeros_like(...)
└── ...
```

实现：

```text
NumpyBackend
CupYBackend
```

然后 Tensor 保存：

```text
data
backend
device
```

这样 Autograd graph 无需关心底层设备。

## 5. 新增 view / stride semantics

这是下一阶段较难但非常重要的方向。

建议增加：

```text
Storage
Tensor.storage_offset
Tensor.shape
Tensor.stride
Tensor.is_contiguous
```

再重新实现：

```text
view
reshape
transpose
contiguous
```

不要直接依赖 NumPy reshape 行为作为最终语义。

## 6. 新增 inplace safety

未来可以给 Tensor 加：

```text
_version
```

每次 inplace 修改：

```text
_version += 1
```

Function forward 保存版本号，backward 时检测版本是否变化。

这能非常直观地解释真实 PyTorch 为什么很多 inplace 操作会触发 autograd 错误。
