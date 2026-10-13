# mini-pytorch v0.11

## 1. Version identity

- **Branch:** `v0.11`
- **Previous version:** `v0.10`
- **Theme:** Graph IR / Eager-to-Graph Capture
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.11.0`
- **Validated regression baseline:** **176/176 tests passed**

v0.11 answers:

> How can eager logical-operator execution be represented as explicit `Value / Node / Graph` data and then replayed through the existing Dispatcher without creating a second numerical framework?

## 2. Core difference from v0.10

v0.10 made replicated gradient synchronization explicit:

```text
local backward → Reducer → ProcessGroup → synchronized grad
```

v0.11 introduces the first compiler-facing representation:

```text
Python Tensor calls
      ↓
Dispatcher
      ├── eager execution
      └── capture observer
              ↓
        Value / Node / Graph
              ↓
       GraphInterpreter
              ↓
          Dispatcher
              ↓
       ordinary eager execution
```

The Graph records logical operator semantics. It does not own kernels, Autograd formulas, backend selection, optimization or code generation.

## 3. Implemented mechanisms

### Explicit IR

Added:

```text
Value
ValueRef
TensorConstant
Node
Graph
```

Value kinds:

```text
input
constant
intermediate
```

Tensor Values carry descriptive metadata:

```text
shape
dtype
device
requires_grad
```

IDs are deterministic within one capture:

```text
%v0, %v1, ...
node0, node1, ...
```

### Stable textual IR

`Graph.format()` renders deterministic human-readable IR. Graph input numerical values are not embedded, so equivalent traces with different input values produce the same textual program when constants/operator structure are unchanged.

### Dispatcher-level capture observer

Capture observes only `Dispatcher.dispatch()`.

Internal:

```text
Autograd → redispatch → CPU/EDU
```

is not recorded as extra nodes.

Therefore one logical eager op produces one IR Node.

### Capture API

Added:

```python
captured = capture(fn, *example_tensor_inputs)
```

which returns:

```text
CapturedGraph
├── graph
└── eager_output
```

The trace run is a normal eager execution plus observation.

### Tensor constants

Pre-existing leaf Tensors not registered as graph inputs are captured by value as `TensorConstant` payloads containing:

```text
data copy
requires_grad
device
```

Replay materializes fresh Tensor constants, so the Graph does not alias later mutation of the original Python Tensor.

### Structured args / kwargs

Node argument specs support:

```text
ValueRef
None
bool/int/float/str
tuple
list
dict
NumPy scalar → Python scalar
```

This preserves operator options such as `axis`, `keepdims`, convolution stride/padding/dilation and implementation choice.

### GraphInterpreter

Added a reference interpreter that:

```text
binds runtime inputs
materializes constants
resolves structured ValueRefs
calls Dispatcher.dispatch for every node
binds node outputs
returns graph outputs
```

It never invokes backend kernels directly.

### Runtime dispatch remains dynamic

The Graph stores logical operator names, not trace-time dispatch keys.

A graph captured under:

```text
NoGrad + CPU
```

can later replay under:

```text
Autograd + CPU
```

when runtime inputs require grad and grad mode is enabled.

This proves:

```text
Graph IR != DispatchKeySet
```

### CPU / EDU replay

Replay through the normal Dispatcher preserves backend routing. EDU capture/replay tests produce EDU outputs and ordinary backward gradients.

### GraphBreak semantics

If a dynamic Tensor is produced by an unsupported/unobserved eager path and later crosses into the captured graph, v0.11 raises `GraphBreakError` instead of silently freezing the trace-time value.

Example unsupported path:

```python
y = x.exp()   # direct Function.apply path in current framework
z = y + x     # captured Dispatcher op
```

`y` is an untracked non-leaf Tensor, so capture breaks explicitly.

A direct unsupported Tensor returned as the captured function output also breaks.

### Current Dispatcher-visible operator surface

The observer can record the current logical operator registry, including:

```text
aten::add.Tensor
aten::mul.Tensor
aten::matmul.default
aten::relu.default
aten::sum.default
aten::conv2d.default
aten::max_pool2d.default
```

## 4. Central correctness proof

For a supported eager function:

```text
fresh values A
   ↓
ordinary eager function
   ↓
output_eager

fresh values B with same data
   ↓
GraphInterpreter(captured_graph)
   ↓
output_graph
```

CI verifies:

```text
output_graph == output_eager
```

Then both outputs execute ordinary backward and CI verifies matching graph-input gradients.

Additional proofs cover:

- matmul replay + gradients;
- conv2d replay with captured Tensor constants;
- EDU replay + backward;
- NoGrad trace followed by Autograd replay;
- replay uses Dispatcher rather than direct kernels.

## 5. New files

```text
mini_pytorch/compiler/__init__.py
mini_pytorch/compiler/ir.py
mini_pytorch/compiler/capture.py
mini_pytorch/compiler/interpreter.py

tests/test_graph_ir.py
docs/GRAPH_IR.md
examples/14_graph_capture.py
```

Major modified files:

```text
mini_pytorch/dispatcher.py
mini_pytorch/__init__.py
pyproject.toml
README.md
MANIFEST.txt
Agents.md
version.md
```

## 6. Preserved invariants

v0.11 preserves all earlier guarantees, including:

- Storage/stride/view/version-counter semantics;
- Dispatcher and redispatch ordering;
- CPU/EDU backend routing;
- Autograd graph lifetime/hooks/custom Function semantics;
- checkpoint deterministic resume;
- DDP global-batch reference equivalence.

Eager execution outside an active capture is unchanged except for a no-op capture-observer check after outer dispatch.

## 7. Explicit non-goals

Frozen v0.11 does not implement:

- Python bytecode/frame tracing;
- arbitrary Python control flow;
- symbolic shapes;
- shape/device/dtype guards;
- graph cache;
- automatic recompilation;
- full Module/Parameter lifting;
- mutation/alias functionalization;
- AOTAutograd;
- graph optimization/fusion;
- lowering/code generation;
- Inductor/Triton.

## 8. Validation

Core implementation and extended architecture tests passed on GitHub Actions:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
Ran 176 tests
OK
```

The exact frozen branch HEAD must remain green; any future v0.11 fix must rerun the complete suite.

## 9. Recommended reading order

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

## 10. Next version

`v0.12` owns **Guards / Graph Cache / Recompile**.

v0.11 asks:

> What is the graph?

v0.12 asks:

> Under what runtime assumptions is the captured graph valid, when can it be reused, and when must a new graph be captured?
