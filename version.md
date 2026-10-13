# mini-pytorch v0.13

## Version identity

- **Branch:** `v0.13`
- **Previous version:** `v0.12`
- **Theme:** AOTAutograd-like Graph Differentiation
- **Status:** active development
- **Package version:** `0.13.0`
- **Starting regression baseline:** **192/192 tests passed**

v0.13 asks:

> Can reverse-mode differentiation itself become an explicit graph program rather than opaque eager Python/NumPy VJP execution?

## Core delta from v0.12

v0.12 established reusable guarded forward Graphs. v0.13 adds an explicit forward/backward ABI:

```text
forward Graph
    ↓
differentiate_graph
    ↓
DifferentiationSpec
    ↓
backward Graph

forward execution
    ├── user output
    └── saved forward Values
               ↓
        backward Graph
        + grad_output
               ↓
         input gradients
```

## Initial supported VJP set

```text
aten::add.Tensor
aten::mul.Tensor
aten::matmul.default   # 2-D
aten::relu.default
aten::sum.default
```

Backward Graph helper logical operators:

```text
aot::unbroadcast.default
aot::sum_backward.default
aot::transpose2d.default
aot::relu_backward.default
```

They execute through Dispatcher on CPU/EDU and are intentionally first-order.

## Required correctness proof

For supported graphs:

```text
eager forward == AOT forward

eager AutogradEngine input gradients
==
explicit backward-Graph input gradients
```

Also required: broadcasting, shared-gradient accumulation, CPU/EDU parity, non-scalar grad seeds, and explicit unsupported-op failures.

## Non-goals

- production AOTAutograd;
- arbitrary operator decompositions;
- mutation functionalization;
- higher-order differentiation;
- symbolic shapes;
- lowering/fusion/code generation.

## Next version

v0.14 will study lowering/fusion once forward and backward graph programs are both explicit and validated.
