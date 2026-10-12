# Learning Path

## 第 1 轮：30 分钟

只运行：

```bash
python examples/01_autograd.py
```

然后阅读：

```text
Tensor.__mul__
Mul.forward
Function.apply
Tensor.backward
Mul.backward
```

目标：解释 `x * y` 为什么既做数值计算，又创建计算图。

## 第 2 轮：60 分钟

阅读：

```text
Add
Mul
Pow
Sum
Mean
unbroadcast
```

然后运行：

```bash
python -m unittest tests.test_autograd -v
python -m unittest tests.test_ops -v
```

目标：解释 broadcasting 和 gradient accumulation。

## 第 3 轮：60 分钟

阅读：

```text
Parameter
Module.__setattr__
named_parameters
state_dict
Linear
Sequential
```

目标：解释为什么 `model.parameters()` 不需要手写参数列表。

## 第 4 轮：60 分钟

阅读：

```text
Optimizer
SGD
Adam
```

重点比较：

```text
SGD:
parameter <- parameter - lr * grad

Momentum:
v <- beta*v + grad
parameter <- parameter - lr*v

Adam:
m <- beta1*m + (1-beta1)*grad
v <- beta2*v + (1-beta2)*grad^2
bias correction
parameter update
```

## 第 5 轮：训练闭环

运行：

```bash
python examples/02_linear_regression.py
python examples/03_mlp_xor.py
```

然后不看源码，尝试口头完整解释：

```python
pred = model(x)
loss = criterion(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

如果五行都能展开到内部机制，就已经掌握了 PyTorch 最核心的训练闭环。

## 第 6 轮：自己扩展

推荐挑战：

1. 实现 `Tanh`；
2. 用 gradcheck 验证；
3. 实现 `Softmax`；
4. 实现 `CrossEntropyLoss`；
5. 实现 `RMSprop`；
6. 实现 `Dropout`；
7. 实现 `Embedding`。

重点不是功能数量，而是保证每个扩展只修改它应该修改的层。
