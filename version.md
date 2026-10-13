# mini-pytorch v0.13

## Version identity

- **Branch:** `v0.13`
- **Previous version:** `v0.12`
- **Theme:** AOTAutograd-like Graph Differentiation
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.13.0`
- **Validated regression baseline:** **206/206 tests passed**

v0.13 answers:

> Once forward execution is explicit Graph data, how can reverse-mode differentiation itself become an explicit graph program instead of remaining opaque eager VJP execution?

## Core delta from v0.12

```text
v0.12
forward Graph + guards/cache

v0.13
forward Graph
    ↓
differentiate_graph
    ↓
DifferentiationSpec
    ↓
explicit saved-value ABI
    ↓
backward Graph
```

Runtime becomes:

```text
inputs
  ↓
forward Graph
  ├── user output
  └── saved forward Values
             ↓
       backward Graph
       + grad_output
             ↓
       input gradients
```

## Implemented components

```text
DifferentiationSpec
AOTContext
AOTForwardResult
AOTProgram
AOTDifferentiationError
differentiate_graph(graph)
aot_compile(fn, *example_inputs)
```

`GraphInterpreter.run_with_values()` exposes selected forward Values so saved tensors are an explicit program interface rather than hidden Python closure state.

## Supported VJP rules

```text
aten::add.Tensor
aten::mul.Tensor
aten::matmul.default   # 2-D
aten::relu.default
aten::sum.default
```

Compiler VJP policy emits normal logical ops plus first-order helper ops:

```text
aot::unbroadcast.default
aot::sum_backward.default
aot::transpose2d.default
aot::relu_backward.default
```

All helper operators are registered for CPU and EDU and execute through Dispatcher.

## Saved forward values

The backward interface explicitly records deterministic forward Value ids needed by VJP rules.

Examples:

```text
mul     → operands
matmul  → operands
relu    → forward input
add     → shape metadata only
sum     → input shape + reduction attributes
```

## Broadcast and reduction semantics

`add` and `mul` gradients use explicit `aot::unbroadcast.default` nodes to reduce broadcasted contributions back to the exact input shape.

`sum` gradients use explicit `aot::sum_backward.default` nodes carrying:

```text
input_shape
axis
keepdims
```

## Shared-value gradient accumulation

If one forward Value receives multiple reverse contributions, v0.13 emits explicit `aten::add.Tensor` nodes in the backward Graph.

This makes accumulation graph data rather than hidden Engine state.

## MatMul boundary

The existing framework MatMul supports 2-D operands, so the AOT rule preserves that exact boundary:

```text
dA = dY @ B.T
dB = A.T @ dY
```

Transposes are explicit `aot::transpose2d.default` nodes.

## First-order boundary

`AOTProgram.backward()` executes the explicit backward Graph under `no_grad`.

Therefore:

```text
backward is explicit Graph data
```

but:

```text
higher-order differentiation is NOT supported
```

Returned gradient Tensors have no `GradFn`.

## Correctness proof

The v0.13 test suite verifies:

```text
AOT forward output == eager forward output
```

and:

```text
explicit backward-Graph gradients
==
eager AutogradEngine gradients
```

Coverage includes:

- add + sum;
- mul broadcasting;
- ReLU;
- 2-D MatMul;
- shared-subgraph accumulation;
- non-grad graph inputs;
- non-scalar explicit grad seeds;
- axis/keepdims reduction;
- CPU/EDU device preservation;
- unsupported-op errors;
- first-order/no-GradFn boundary.

## Important files

```text
mini_pytorch/compiler/aot.py
mini_pytorch/aot_kernels.py
mini_pytorch/compiler/interpreter.py
mini_pytorch/operator_registry.py

tests/test_aot_autograd.py
docs/AOT_AUTOGRAD.md
examples/16_aot_autograd.py
```

Recommended reading order:

```text
compiler/aot.py
    ↓
aot_kernels.py
    ↓
operator_registry.py
    ↓
compiler/interpreter.py
    ↓
test_aot_autograd.py
    ↓
AOT_AUTOGRAD.md
    ↓
16_aot_autograd.py
```

## Non-goals

v0.13 does not implement:

- production PyTorch AOTAutograd;
- arbitrary mutation functionalization;
- all-operator decompositions/VJPs;
- symbolic shapes;
- higher-order gradients;
- compiler lowering/fusion/code generation.

## Next version

v0.14 should study **Lowering / Fusion / Inductor-like Concepts** now that both forward and backward are explicit graph programs.
