# mini-pytorch v0.12 — Guards / Graph Cache / Recompile

> v0.11 回答“eager execution 如何变成 Graph IR”；v0.12 回答“这个 Graph 在什么条件下还能安全复用”。

**Package version:** `0.12.0`

## 核心增量

```text
v0.11
capture(fn, example_inputs)
      ↓
Graph
      ↓
GraphInterpreter

v0.12
runtime inputs
      ↓
InputSignature
      ↓
GuardSet
      ↓
GraphCache
 ┌──────┴──────┐
 HIT           MISS
  │             │
 replay       capture
                ↓
             new entry
```

一个 Python callable 不再只对应一个 Graph，而可能对应多个 specialization：

```text
fn
├── graph0: shape=(2,4), float64, cpu, grad=True
├── graph1: shape=(8,4), float64, cpu, grad=True
└── graph2: shape=(2,4), float64, edu, grad=True
```

## v0.12 guard fields

当前保守 specialization 精确 guard：

```text
input count
shape
dtype
device
requires_grad
grad mode
```

不同字段意味着 cache miss，而不是偷偷复用不兼容 Graph。

## Quick start

```python
import numpy as np
from mini_pytorch import Tensor
from mini_pytorch.compiler import compile_graph

compiled = compile_graph(lambda x: (x + x).sum())

compiled(Tensor(np.ones((2, 3))))   # MISS -> compile graph0
compiled(Tensor(np.ones((2, 3))))   # HIT graph0
compiled(Tensor(np.ones((8, 3))))   # MISS -> compile graph1
compiled(Tensor(np.ones((2, 3))))   # HIT graph0

print(compiled.stats())
print(compiled.explain(Tensor(np.ones((4, 3)))).format())
```

期望核心统计：

```text
compile_count = 2
recompile_count = 1
cache_entries = 2
hit_count = 2
miss_count = 2
```

运行完整示例：

```bash
python examples/15_graph_cache.py
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 重要文件

```text
mini_pytorch/compiler/signature.py   # TensorSignature / InputSignature
mini_pytorch/compiler/guards.py      # inspectable guards
mini_pytorch/compiler/cache.py       # guarded multi-entry cache
mini_pytorch/compiler/compiled.py    # CompiledFunction / compile_graph

tests/test_graph_cache.py
docs/GRAPH_CACHE.md
examples/15_graph_cache.py
```

推荐阅读：

```text
signature.py
    ↓
guards.py
    ↓
cache.py
    ↓
compiled.py
    ↓
test_graph_cache.py
    ↓
GRAPH_CACHE.md
    ↓
15_graph_cache.py
```

## Guard failure 不是 runtime error

```text
entry guard FAIL
      ↓
normal cache miss
      ↓
capture another specialization
```

真正异常用于 malformed state 或 v0.11 已定义的 unsupported GraphBreak。

## GraphBreak policy

如果 capture 遇到不支持的数据流：

```text
GraphBreakError
      ↓
no cache entry
compile_count unchanged
graph_break_count += 1
```

不会自动把 Python 函数再跑一遍，因为重复执行可能复制副作用。

## Constants policy

v0.11 的 closure Tensor constant 继续采用 capture-by-value：

```text
closure Tensor
      ↓ compile
frozen TensorConstant in Graph
```

cache hit 使用这个 snapshot。希望随调用变化的值必须作为显式 graph input。

## 明确非目标

v0.12 不实现：

- symbolic shapes；
- Python bytecode/frame interception；
- production Dynamo guard system；
- automatic graph partitioning；
- AOTAutograd；
- kernel lowering/fusion。

## 下一版本

`v0.13`：**AOTAutograd-like graph differentiation**。
