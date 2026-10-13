# v0.13 — AOTAutograd-like Graph Differentiation

v0.13 asks one question:

> Once forward execution is explicit Graph data, can reverse-mode differentiation also become an explicit graph program?

This version is intentionally educational. It does **not** reproduce production PyTorch AOTAutograd, functionalization, decompositions, or higher-order differentiation.

## 1. Before v0.13

```text
forward Graph
      ↓
GraphInterpreter
      ↓
Tensor / GradFn
      ↓
AutogradEngine.backward()
      ↓
Python / NumPy VJP formulas
```

The forward program was visible, but backward remained an eager runtime mechanism.

## 2. v0.13 architecture

```text
                  forward Graph
                       │
                       ▼
              differentiate_graph
                       │
                       ▼
              DifferentiationSpec
             ┌─────────┴─────────┐
             ▼                   ▼
      saved Value ids       grad input ids
             │                   │
             └─────────┬─────────┘
                       ▼
                 backward Graph

runtime forward:
inputs → forward Graph → user output
                  └──→ saved forward Values

runtime backward:
saved Values + grad_output
           ↓
     backward Graph
           ↓
    input gradients
```

The eager `AutogradEngine` remains the reference oracle used by tests.

## 3. Explicit forward/backward ABI

`DifferentiationSpec` records:

```text
forward_input_ids     # graph inputs whose gradients are returned
forward_output_id     # differentiated output
saved_value_ids       # forward Values crossing into backward
```

Saved values are deterministic and inspectable. They are not hidden in Python closures or eager `Context` objects.

`AOTProgram.forward()` returns:

```text
AOTForwardResult
├── output
└── context
    ├── saved_values
    └── output
```

`AOTProgram.backward(context, grad_output)` consumes that explicit state.

## 4. Why saved values are necessary

Different VJPs need different forward information:

```text
add       → only gradient shape information
mul       → original operands
matmul    → original operands
relu      → original forward input / mask information
sum       → original input shape + reduction attributes
```

v0.13 computes a deterministic saved-value set from the reachable forward graph.

## 5. Supported first-wave VJPs

```text
aten::add.Tensor
aten::mul.Tensor
aten::matmul.default   # 2-D only
aten::relu.default
aten::sum.default
```

The implementation walks forward nodes in reverse order and maintains:

```text
forward Value id
      ↓
current backward gradient ValueRef
```

When a Value receives multiple contributions, the backward Graph emits an explicit `aten::add.Tensor` node.

## 6. Broadcasting

For a broadcasted forward expression:

```text
x.shape = (2, 3)
y.shape = (3,)
z = x * y
```

raw reverse contributions initially have the broadcasted output shape. v0.13 therefore emits:

```text
aot::unbroadcast.default
```

so each gradient returns to the exact forward-input shape.

This is graph data, not a hidden post-processing step.

## 7. Reduction backward

For:

```python
x.sum(axis=1, keepdims=False)
```

backward needs to reconstruct reduced dimensions and broadcast the seed to `x.shape`.

The backward Graph emits:

```text
aot::sum_backward.default
```

with explicit:

```text
input_shape
axis
keepdims
```

## 8. ReLU backward

ReLU differentiation needs forward sign information:

```text
dx = grad_output * (x > 0)
```

The forward input is therefore a saved Value. The backward Graph contains:

```text
aot::relu_backward.default(grad, saved_x)
```

## 9. MatMul backward

The existing mini-pytorch MatMul supports 2-D operands, so v0.13 preserves the same boundary:

```text
Y = A @ B

dA = dY @ B.T
dB = A.T @ dY
```

The backward Graph makes transpose operations explicit with:

```text
aot::transpose2d.default
```

and then reuses the normal logical `aten::matmul.default` operator.

## 10. AOT helper operators are not a second backend

The helper logical operators are:

```text
aot::unbroadcast.default
aot::sum_backward.default
aot::transpose2d.default
aot::relu_backward.default
```

They follow the same execution path as other logical ops:

```text
backward Graph
      ↓
GraphInterpreter
      ↓
Dispatcher
      ↓
CPU / EDU backend kernel
```

The compiler owns VJP *policy*. Pure NumPy helper kernels own numerical mechanism.

## 11. Why backward runs under `no_grad`

v0.13 constructs an explicit backward Graph, but the execution of that Graph is intentionally first-order:

```text
backward Graph exists
      ≠
backward execution builds another differentiable graph
```

`AOTProgram.backward()` runs under `no_grad`, so returned gradients have no `GradFn`.

True higher-order differentiation would require differentiating or tracing the backward Graph itself and providing differentiable rules for helper ops. That is not claimed here.

## 12. CPU / EDU semantics

CPU and EDU versions of the same forward program generate the same logical backward operator sequence, but Graph metadata correctly preserves different devices.

```text
same logical structure
+
different device metadata
```

is expected and tested.

## 13. Error boundaries

v0.13 fails explicitly when:

- the forward Graph has no single Tensor output;
- no graph input requires grad;
- a reachable operator has no VJP rule;
- MatMul leaves the existing 2-D boundary;
- non-scalar output omits `grad_output`;
- `grad_output` shape/device mismatches the forward output.

Unsupported operators do not silently fall back to eager backward.

## 14. Central correctness proof

For every supported test graph:

```text
AOT forward output == eager forward output
```

and:

```text
explicit backward-Graph gradients
==
eager AutogradEngine gradients
```

Coverage includes broadcast, reduction axes, ReLU, MatMul, shared subgraphs, non-grad inputs and CPU/EDU.

## 15. Read next

```text
mini_pytorch/compiler/aot.py
      ↓
mini_pytorch/aot_kernels.py
      ↓
mini_pytorch/operator_registry.py
      ↓
mini_pytorch/compiler/interpreter.py
      ↓
tests/test_aot_autograd.py
      ↓
examples/16_aot_autograd.py
```

The next milestone, v0.14, can finally treat both forward and backward as explicit programs and study lowering/fusion transformations.
