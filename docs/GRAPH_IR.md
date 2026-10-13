# Graph IR / Eager-to-Graph Capture

mini-pytorch v0.11 introduces the first compiler-facing representation in the project.

The goal is **not** performance. The goal is to make one previously invisible fact explicit:

> An eager program is a sequence of logical operator applications. A compiler needs that sequence represented as data before it can guard, cache, transform, differentiate ahead of time, lower, or fuse it.

## 1. Architectural position

Before v0.11:

```text
Python Tensor API
      ↓
Dispatcher
      ↓
Autograd / NoGrad / backend routing
      ↓
CPU / EDU numerical kernel
      ↓
Tensor result
```

v0.11 adds an observer at the **outer logical dispatch boundary**:

```text
Python Tensor API
      ↓
Dispatcher.dispatch(operator, ...)
      ├──────────────────────────────┐
      ▼                              ▼
ordinary eager execution       capture observer
      │                              │
      ▼                              ▼
Tensor result                  Value / Node / Graph
```

The observer runs after the eager result exists. It records semantics; it does not replace execution.

## 2. Why capture at Dispatcher instead of kernels?

One eager multiplication with gradients may execute:

```text
aten::mul.Tensor
      ↓
DispatchKeySet {Autograd, CPU}
      ↓
Autograd wrapper
      ↓
redispatch {CPU}
      ↓
CPU mul kernel
```

The graph should contain **one** node:

```text
aten::mul.Tensor
```

not:

```text
AutogradMul
CPU_Mul
```

Why?

Because Autograd/CPU/EDU are runtime policy/backend decisions. The logical program says only:

```text
multiply these Tensor values
```

That is why the observer lives in `Dispatcher.dispatch()` and **not** in `redispatch()`.

## 3. IR objects

The v0.11 IR is deliberately small:

```text
Graph
├── Value
├── Node
├── inputs
└── outputs
```

### Value

A `Value` is one data edge.

Kinds:

```text
input
constant
intermediate
```

Each Tensor Value carries metadata:

```text
shape
dtype
device
requires_grad
```

Metadata is descriptive in v0.11. It is **not yet a guard**.

### Node

A Node records:

```text
node id
logical operator qualified name
structured positional args
structured keyword args
structured outputs
```

Tensor edges are represented by stable `ValueRef(value_id)` objects rather than Python object addresses.

### Graph

A Graph owns deterministic integer IDs:

```text
%v0
%v1
%v2
...

node0
node1
...
```

This makes the textual IR stable across captures that execute the same logical program with different graph-input values.

## 4. Graph inputs

Example:

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
```

`x` and `y` become Graph input Values.

Their numerical contents are **not** embedded into the IR.

Therefore a replay can bind fresh runtime Tensors:

```python
GraphInterpreter(captured.graph)(new_x, new_y)
```

v0.11 accepts positional Tensor graph inputs only. Python argument specialization belongs to v0.12 guards/cache work.

## 5. Tensor constants

A Tensor used by the function but not declared as a graph input may be captured by value if it is a leaf / pre-existing Tensor:

```python
scale = Tensor([2.0, 3.0])

graph = capture(
    lambda x: x * scale,
    Tensor([1.0, 1.0]),
).graph
```

`scale` becomes a `constant` Value.

The IR stores a copy of:

```text
data
requires_grad
device
```

Replay materializes a fresh Tensor constant. Mutating the original `scale` after capture does not mutate the Graph.

This is intentionally simple. Full Module/Parameter lifting and state mutation semantics are not claimed in v0.11.

## 6. Python literals

Structured Python literals are stored inline in node argument specs:

```text
None
bool
int
float
str
tuple/list/dict of supported values
```

Example:

```python
x.sum(axis=1, keepdims=True)
```

records:

```text
kwargs={'axis': 1, 'keepdims': True}
```

## 7. Intermediate Values

Suppose:

```python
z = x * y
q = z + x
r = q.relu()
out = r.sum()
```

The capture map connects the actual eager Tensor returned by each Dispatcher call to one IR Value:

```text
%x, %y
  ↓
node0 mul → %z
  ↓
node1 add → %q
  ↓
node2 relu → %r
  ↓
node3 sum → %out
```

Python object identity exists only during capture to answer:

> Have I already assigned this eager Tensor an IR Value?

It is not part of the stored graph representation.

## 8. Stable textual IR

`Graph.format()` returns a deterministic teaching format.

Conceptually:

```text
graph(%v0: Tensor(...), %v1: Tensor(...)):
  %v2 = aten::mul.Tensor(args=(%v0, %v1), kwargs={})  # node0
  %v3 = aten::add.Tensor(args=(%v2, %v0), kwargs={})  # node1
  %v4 = aten::relu.default(args=(%v3,), kwargs={})     # node2
  %v5 = aten::sum.default(args=(%v4,), kwargs={...})   # node3
  return %v5
```

The exact syntax is educational and not intended as a stable external file format.

## 9. GraphInterpreter

Replay is intentionally boring:

```text
bind graph inputs
      ↓
materialize constants
      ↓
for node in graph.nodes:
    resolve ValueRefs
    ↓
    Dispatcher.dispatch(node.operator, ...)
    ↓
    bind result to output ValueRefs
      ↓
resolve graph outputs
```

The interpreter does **not** call CPU/EDU kernels directly.

That preserves all earlier architecture:

```text
GraphInterpreter
      ↓
Dispatcher
      ↓
Autograd / NoGrad
      ↓
CPU / EDU
      ↓
ordinary Tensor / GradFn
```

## 10. Why replay can still backward

A replayed output is an ordinary Tensor produced by ordinary Dispatcher execution.

Therefore:

```python
out = GraphInterpreter(graph)(x, y)
out.backward()
```

uses the normal:

```text
GradFn / Context
      ↓
AutogradEngine
```

v0.11 does not capture a backward graph and does not implement AOTAutograd.

The central proof only checks that replaying the forward program reconstructs an eager Autograd graph with equivalent input gradients.

## 11. Dispatch policy is not baked into IR

This distinction is essential.

A graph may be captured while `no_grad()` is active:

```text
trace execution
DispatchKeySet {NoGrad, CPU}
```

The IR still stores:

```text
aten::mul.Tensor
aten::sum.default
```

Later replay with a `requires_grad=True` input and grad mode enabled recomputes:

```text
DispatchKeySet {Autograd, CPU}
```

and produces a differentiable output.

So:

```text
Graph IR = logical program
DispatchKeySet = runtime execution policy
```

They are separate layers.

## 12. CPU / EDU

Graph replay uses runtime Tensor placement.

A graph captured/replayed with EDU inputs routes back through:

```text
GraphInterpreter
      ↓
Dispatcher
      ↓
EDU DispatchKey
      ↓
EDU backend registration
```

Tensor constants currently retain their captured device. A CPU constant mixed with an EDU runtime input therefore correctly triggers the framework's existing mixed-device error.

Automatic constant/device specialization belongs to later compiler/runtime work.

## 13. GraphBreak

Silent miscapture is worse than an explicit unsupported boundary.

Example:

```python
y = x.exp()      # direct Function.apply path; not Dispatcher-observed yet
z = y + x        # Dispatcher-visible
```

When capture reaches `z = y + x`, `y` is an untracked non-leaf Tensor.

Treating `y` as a constant would freeze the trace-time value and produce the wrong program.

Therefore v0.11 raises:

```text
GraphBreakError
```

Similarly, returning `x.exp()` directly from the captured function breaks when graph outputs are finalized.

This establishes a key compiler principle:

> Unsupported dynamic dataflow must be explicit; it must not silently become a constant.

## 14. Current supported operator surface

Capture observes any operator that goes through the existing Dispatcher, currently including:

```text
aten::add.Tensor
aten::mul.Tensor
aten::matmul.default
aten::relu.default
aten::sum.default
aten::conv2d.default
aten::max_pool2d.default
```

Operators still implemented only through direct `Function.apply` paths are not yet graph-capturable when their dynamic result crosses the graph boundary.

## 15. What v0.11 intentionally does not solve

No shape guards:

```text
captured shape metadata
≠
shape specialization contract
```

No graph cache:

```text
same Python function called again
≠
automatic cached graph reuse
```

No Python control-flow capture:

```python
if x.item() > 0:
    ...
```

No Module lifting or mutable state analysis.

No AOT backward graph.

No fusion/lowering/code generation.

These omissions are the reason later versions exist.

## 16. Version bridge

```text
v0.11 Graph IR
      ↓
v0.12 Guards + Graph Cache
      ↓
v0.13 AOTAutograd-like
      ↓
v0.14 Lowering + Fusion
      ↓
v0.15 Runtime / Performance
```

v0.11 answers:

> What is the explicit graph?

v0.12 will answer:

> Under what runtime assumptions is this graph still valid, and when can it be reused?
