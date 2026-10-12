# Architecture

## 1. 总览

```text
User API
  │
  ├── Tensor operators
  │      ↓
  │   Function.apply
  │      ↓
  │   NumPy forward
  │      ↓
  │   GradFn graph
  │
  ├── nn.Module
  │      ↓
  │   Parameter registry
  │
  └── Optimizer
         ↓
      Parameter update
```

## 2. Autograd 数据结构

每个需要梯度的非叶子 Tensor 持有一个 `GradFn`：

```text
Tensor
├── data
├── requires_grad
├── grad
└── _grad_fn
      ├── function
      ├── parents
      └── ctx
```

### Function

一个 Operator 的数学行为：

```text
forward(ctx, inputs...) -> output
backward(ctx, grad_output) -> grad_inputs...
```

### Context

只保存 backward 真正需要的中间结果。

例如乘法：

```text
z = x * y
```

backward 需要 x 和 y，因此 forward 使用：

```python
ctx.save_for_backward(x, y)
```

## 3. 为什么 graph 挂在输出 Tensor 上

执行：

```python
z = x * y
```

得到：

```text
x ----\
       MulBackward -> z
y ----/
```

下一步：

```python
q = z + x
```

变为：

```text
x ----\
       MulBackward -> z --\
y ----/                  AddBackward -> q
x ----------------------/
```

因此 eager mode 中，图是随着 Python 程序实际执行动态产生的。

## 4. backward 为什么先拓扑排序

如果一个节点被重复使用：

```text
       ┌---- path A ----┐
x -----|                +---- loss
       └---- path B ----┘
```

必须先收集来自所有下游分支的梯度，再继续向更上游传播。

因此 DAG 的 reverse topological traversal 是自然方案。

## 5. Module 层

`Module` 负责组织参数，而不是负责求导。

```text
Module
├── _parameters
├── _modules
├── _buffers
└── training
```

Autograd 不关心参数属于哪个 Module；它只认识 Tensor graph。

Optimizer 也不关心模型结构；它只拿到 Parameter 列表。

这种解耦非常关键：

```text
Autograd ≠ Module ≠ Optimizer
```

## 6. Optimizer 层

`Optimizer` 不做求导。

它只消费已经由 backward 得到的：

```text
parameter.grad
```

然后修改：

```text
parameter.data
```

这解释了为什么训练顺序必须是：

```text
forward
loss
zero_grad
backward
step
```

## 7. Parameter Groups

Optimizer 保存：

```text
param_groups = [
  {params: [...], lr: ..., ...},
  {params: [...], lr: ..., ...}
]
```

因此超参数并不是一定“一个优化器只有一套”。

这是 fine-tuning、分层学习率等功能的基础。
