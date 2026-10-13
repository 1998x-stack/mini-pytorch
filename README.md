# mini-pytorch v0.11 — Graph IR / Eager-to-Graph Capture

> 把 eager Tensor 执行第一次变成显式、可检查、可重放的 `Value / Node / Graph`。

**Package version:** `0.11.0`

**Validated target:** full regression + Graph IR capture/replay tests

## 1. 本版本核心问题

到 v0.10，mini-pytorch 已经有：

```text
Tensor
  ↓
Dispatcher
  ↓
Autograd / Backend
  ↓
Kernel
```

但一次 Python eager 执行完成后，框架并没有一个独立的数据结构描述：

```text
到底执行了哪些 logical operators？
它们之间的数据依赖是什么？
如果给一组新的输入，能不能按同样的 operator graph 重放？
```

v0.11 新增：

```text
Python eager function
      ↓
Dispatcher
      ├── normal eager execution
      └── capture observer
              ↓
          Graph IR
      ┌───────┼───────┐
      ▼       ▼       ▼
    Value    Node    outputs
              │
              ▼
      GraphInterpreter
              │
              ▼
          Dispatcher
```

关键原则：

> **Graph 不实现数学；Graph 只表示 logical program。**

## 2. 相比 v0.10 的变化

v0.10：

```text
多个 replica 的 local gradients
        ↓
Reducer / ProcessGroup
        ↓
同步梯度
```

v0.11：

```text
一次 eager operator execution
        ↓
显式 Graph IR
        ↓
再次解释执行
```

项目从 eager runtime 正式进入 compiler representation 路线。

## 3. 新增目录

```text
mini_pytorch/compiler/
├── __init__.py
├── ir.py
├── capture.py
└── interpreter.py
```

新增：

```text
tests/test_graph_ir.py
docs/GRAPH_IR.md
examples/14_graph_capture.py
```

## 4. Value / Node / Graph

### Value

代表 graph 中的一条数据边：

```text
input
constant
intermediate
```

Tensor metadata：

```text
shape
dtype
device
requires_grad
```

### Node

代表一次 logical operator application：

```text
operator
args
kwargs
outputs
```

例如：

```text
aten::mul.Tensor
aten::add.Tensor
aten::relu.default
aten::sum.default
```

### Graph

拥有：

```text
inputs
values
nodes
outputs
```

Value 和 Node ID 按 capture 顺序稳定增长：

```text
%v0, %v1, %v2 ...
node0, node1, node2 ...
```

## 5. Capture API

```python
from mini_pytorch import Tensor
from mini_pytorch.compiler import capture

x = Tensor([1.0, 2.0], requires_grad=True)
y = Tensor([3.0, 4.0], requires_grad=True)

captured = capture(
    lambda a, b: ((a * b) + a).relu().sum(),
    x,
    y,
)

print(captured.graph.format())
```

capture 本身仍然执行普通 eager forward。

因此返回：

```text
CapturedGraph
├── graph
└── eager_output
```

不是 symbolic-only fake execution。

## 6. 为什么 observer 放在 Dispatcher.dispatch

一次带梯度的乘法实际可能走：

```text
aten::mul.Tensor
      ↓
Autograd dispatch
      ↓
redispatch
      ↓
CPU / EDU
```

IR 只应看到：

```text
aten::mul.Tensor
```

所以 capture hook 只放在：

```text
Dispatcher.dispatch
```

而不放在：

```text
Dispatcher.redispatch
```

从而避免：

```text
一个 logical op
被重复记录成多节点
```

## 7. GraphInterpreter

```python
from mini_pytorch.compiler import GraphInterpreter

out = GraphInterpreter(captured.graph)(new_x, new_y)
```

执行过程：

```text
bind inputs
    ↓
materialize constants
    ↓
Node 0 → Dispatcher
    ↓
Node 1 → Dispatcher
    ↓
...
    ↓
resolve graph output
```

Interpreter **不会直接调用 CPU/EDU kernel**。

因此 replay 自动继承：

```text
Autograd
NoGrad
CPU / EDU
DispatchKey
GradFn
AutogradEngine
```

## 8. Forward + Backward equivalence

v0.11 的中央 correctness proof：

```text
fresh inputs A
    ↓
eager Python function
    ↓
output_eager

fresh inputs B with same values
    ↓
GraphInterpreter
    ↓
output_graph
```

要求：

```text
output_eager == output_graph
```

随后分别：

```text
output_eager.backward()
output_graph.backward()
```

要求 graph inputs 的 gradients 同样一致。

这证明 replay 没有绕开原来的 Autograd 架构。

## 9. Dispatch policy 不固化在 Graph 中

如果 trace 在：

```python
with no_grad():
    captured = capture(...)
```

trace 时可能是：

```text
{NoGrad, CPU}
```

Graph 中仍只保存：

```text
aten::mul.Tensor
```

以后用 `requires_grad=True` Tensor replay：

```text
Dispatcher
→ {Autograd, CPU}
```

仍然会构造 GradFn。

因此：

```text
IR = logical semantics
DispatchKeySet = runtime policy
```

## 10. Tensor constant

闭包中的 leaf Tensor 可以被 capture by value：

```python
scale = Tensor([2.0, 3.0])

graph = capture(
    lambda x: x * scale,
    Tensor([1.0, 1.0]),
).graph
```

`scale` 变成：

```text
constant Value
```

Graph 保存独立数据副本，不继续 alias 原对象。

完整 Module/Parameter lifting 不属于 v0.11。

## 11. GraphBreak

以下路径目前不是 Dispatcher logical op：

```python
y = x.exp()
```

如果继续：

```python
z = y + x
```

capture 会发现 `y` 是一个未记录的 dynamic non-leaf Tensor。

如果把它偷偷当 constant：

```text
trace-time y
被永久冻结
```

Graph 就错了。

所以 v0.11 明确抛：

```text
GraphBreakError
```

原则：

> **宁愿显式 graph break，也不能静默 miscapture。**

## 12. 当前可捕获 logical ops

凡是走现有 Dispatcher 的 operator 都可以被 observer 记录，目前包括：

```text
aten::add.Tensor
aten::mul.Tensor
aten::matmul.default
aten::relu.default
aten::sum.default
aten::conv2d.default
aten::max_pool2d.default
```

`exp/log` 等仍为 direct Function path，跨入 captured dataflow 时会 graph break。

## 13. CPU / EDU

Replay 重新走 Dispatcher，因此 EDU graph 仍走 EDU registration。

测试覆盖：

```text
capture on EDU
      ↓
GraphInterpreter
      ↓
EDU output
      ↓
backward
```

## 14. 推荐阅读顺序

```text
mini_pytorch/compiler/ir.py
        ↓
mini_pytorch/compiler/capture.py
        ↓
mini_pytorch/dispatcher.py
        ↓
mini_pytorch/compiler/interpreter.py
        ↓
tests/test_graph_ir.py
        ↓
docs/GRAPH_IR.md
        ↓
examples/14_graph_capture.py
```

## 15. 快速运行

```bash
pip install -e .
python -m unittest discover -s tests -v
python examples/14_graph_capture.py
```

## 16. Explicit non-goals

v0.11 不实现：

- Python bytecode/frame tracing；
- dynamic control-flow capture；
- shape/device guards；
- graph cache；
- automatic recompile；
- full Module/Parameter lifting；
- AOT backward graph；
- graph optimization/fusion；
- lowering/codegen；
- Inductor/Triton。

## 17. 下一版本

`v0.12`：**Guards / Graph Cache / Recompile**。

v0.11 回答：

```text
Graph 是什么？
```

v0.12 将回答：

```text
这个 Graph 在什么条件下仍然有效？
什么时候 cache hit？
什么时候必须重新 capture / compile？
```
