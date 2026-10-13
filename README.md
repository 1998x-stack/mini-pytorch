# mini-pytorch v0.13 — AOTAutograd-like Graph Differentiation

> This branch is the historical learning snapshot where **backward itself becomes explicit Graph data**.

**Previous:** `v0.12` Guards / Graph Cache / Recompile  
**Current package:** `0.13.0`  
**Validated release-candidate baseline:** `206 tests`  
**Next:** `v0.14` Lowering / Fusion

## 1. What changed

v0.12 could capture and safely reuse forward Graph specializations, but gradients still came from the eager `AutogradEngine`.

v0.13 adds:

```text
forward Graph
      ↓
differentiate_graph
      ↓
DifferentiationSpec
      ↓
backward Graph
```

At runtime:

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

## 2. Main APIs

```python
from mini_pytorch.compiler import aot_compile, differentiate_graph

program = aot_compile(fn, x_example, y_example)

result = program.forward(x, y)
grads = program.backward(result.context)

print(program.forward_graph.format())
print(program.backward_graph.format())
print(program.spec.saved_value_ids)
```

## 3. Supported VJPs

```text
add
mul
matmul (2-D)
relu
sum
```

Backward Graph helper ops:

```text
aot::unbroadcast.default
aot::sum_backward.default
aot::transpose2d.default
aot::relu_backward.default
```

All replay through the normal Dispatcher and preserve CPU/EDU placement.

## 4. Important files

```text
mini_pytorch/compiler/aot.py
mini_pytorch/aot_kernels.py
mini_pytorch/compiler/interpreter.py
mini_pytorch/operator_registry.py

tests/test_aot_autograd.py
docs/AOT_AUTOGRAD.md
examples/16_aot_autograd.py
```

Recommended order:

```text
aot.py
  ↓
aot_kernels.py
  ↓
operator_registry.py
  ↓
interpreter.py
  ↓
test_aot_autograd.py
  ↓
AOT_AUTOGRAD.md
```

## 5. Central correctness test

```text
eager forward == AOT forward

AutogradEngine gradients
==
backward-Graph gradients
```

The suite also verifies broadcast reduction, axis/keepdims sum backward, shared-gradient accumulation, explicit non-scalar grad seeds, unsupported-op failures and CPU/EDU parity.

## 6. First-order boundary

Backward is now a Graph, but v0.13 intentionally executes it under `no_grad`.

Therefore:

```text
explicit backward Graph
≠
higher-order Autograd
```

The branch makes this limitation explicit rather than pretending `create_graph=True` works.

## 7. Quick run

```bash
python examples/16_aot_autograd.py
python -m unittest discover -s tests -v
```

## 8. Non-goals

Not in v0.13:

- production AOTAutograd;
- arbitrary mutation functionalization;
- all-operator VJP coverage;
- symbolic shapes;
- higher-order gradients;
- lowering/fusion/code generation.

## 9. Next

v0.14 can now work on explicit forward/backward programs and study normalization, dead-value elimination, constant folding, pointwise fusion and lowered execution.
