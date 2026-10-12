# mini-pytorch v0.5 — Dispatcher Deep Dive

## 1. The problem

A Tensor method should express a logical operation, not contain every execution policy.

Bad long-term architecture:

```text
Tensor.add
├── CPU branch
├── CUDA branch
├── Autograd branch
├── no_grad branch
├── autocast branch
└── future compiler/subclass branches
```

v0.5 introduces a routing layer:

```text
Tensor API
  ↓
logical operator
  ↓
Dispatcher
  ↓
DispatchKeySet
  ↓
implementation
```

## 2. OperatorSchema

An operator schema is the identity and contract of a logical operator.

Example:

```text
namespace = aten
name      = add
overload  = Tensor
signature = (Tensor self, Tensor other) -> Tensor
```

Canonical name:

```text
aten::add.Tensor
```

The schema does not say how CPU, Autograd or a future CUDA backend computes the result.

## 3. Kernel registration

One logical operator can have multiple registered implementations:

```text
aten::add.Tensor
├── Autograd → wrapper
└── CPU      → NumPy kernel
```

The registry therefore maps:

```text
(operator, DispatchKey) -> callable
```

This is different from:

```text
operator -> one function
```

because one API can be implemented differently under different execution modes.

## 4. DispatchKeySet

A call derives a set of relevant keys.

### Normal non-grad Tensor

```text
{CPU}
```

### requires-grad Tensor

```text
{Autograd, CPU}
```

### inside no_grad()

```text
{NoGrad, CPU}
```

The educational priority order is:

```text
Autograd > NoGrad > CUDA > CPU
```

`CUDA` is intentionally reserved but not implemented in v0.5.

## 5. Dispatch

Given:

```python
x + y
```

Tensor does:

```text
call("aten::add.Tensor", x, y)
```

Dispatcher then:

1. resolves the schema;
2. infers the DispatchKeySet;
3. selects the highest-priority key;
4. finds a kernel or fallback;
5. invokes it.

## 6. Why Autograd has higher priority than CPU

Suppose:

```python
x.requires_grad = True
y.requires_grad = False
z = x * y
```

The keyset is:

```text
{Autograd, CPU}
```

If CPU ran first, the result would only contain numerical data. No dynamic graph node would be created.

Instead:

```text
Autograd selected
       ↓
prepare backward Context
       ↓
redispatch without Autograd
       ↓
CPU numerical kernel
       ↓
attach GradFn
```

This makes Autograd a wrapper around backend execution.

## 7. Redispatch

Redispatch means:

> invoke the same logical operator again after removing the dispatch key already handled by the current wrapper.

For Autograd:

```text
start: {Autograd, CPU}
          ↓
select Autograd
          ↓
remove Autograd
          ↓
redispatch {CPU}
          ↓
CPU kernel
```

Why not just call `x * y` inside the Autograd wrapper?

Because that would reconstruct:

```text
{Autograd, CPU}
```

and select Autograd again, causing recursion.

## 8. NoGrad as fallback

Inside:

```python
with no_grad():
    y = x + x
```

Dispatcher sees:

```text
{NoGrad, CPU}
```

v0.5 uses a NoGrad fallback rather than a separate numerical kernel:

```text
NoGrad fallback
      ↓
remove NoGrad
      ↓
redispatch {CPU}
      ↓
CPU add
```

This demonstrates an important idea:

> execution policy can change without duplicating backend mathematics.

## 9. CPU kernels

`mini_pytorch/kernels.py` contains pure NumPy implementations such as:

```text
add
mul
matmul
relu
sum
```

They intentionally do not know about:

- `requires_grad`
- `GradFn`
- `no_grad()`
- `Module`
- `Optimizer`

That separation is the architectural point of v0.5.

## 10. Built-in operator registry

`mini_pytorch/operator_registry.py` defines the first five dispatched operators:

```text
aten::add.Tensor
aten::mul.Tensor
aten::matmul.default
aten::relu.default
aten::sum.default
```

For each operator, CPU and Autograd implementations are registered. NoGrad is handled by a dispatcher fallback.

## 11. Introspection

```python
from mini_pytorch import Tensor, dispatcher

x = Tensor([1.0], requires_grad=True)
y = Tensor([2.0])

decision = dispatcher.explain("aten::add.Tensor", x, y)
print(decision.keyset)
print(decision.selected_key)
print(decision.implementation)
```

This makes routing visible instead of implicit.

## 12. Error behavior

v0.5 intentionally fails clearly for:

- duplicate schema definitions;
- kernels registered for undefined operators;
- duplicate `(operator, DispatchKey)` kernels;
- unknown operators;
- unsupported high-priority keys;
- empty dispatch key sets.

For example, manually forcing:

```text
{CUDA, CPU}
```

for an operator with no CUDA kernel raises an error rather than silently pretending CPU is CUDA.

## 13. Mapping to real PyTorch

The conceptual mapping is:

| mini-pytorch | Real PyTorch concept |
|---|---|
| `OperatorSchema` | operator schema / operator handle |
| `Dispatcher` | c10 Dispatcher |
| `DispatchKey` | DispatchKey |
| `DispatchKeySet` | DispatchKeySet |
| CPU kernel | backend kernel |
| Autograd wrapper | autograd dispatch behavior |
| redispatch | redispatch below current key |
| `Library` | registration facade inspired by `torch.library` |

Real PyTorch contains many more keys, aliases, fallback rules, backend/device concerns and boxed/unboxed calling conventions. v0.5 intentionally keeps only the mental model.

## 14. Why this version comes before real device support

Without Dispatcher:

```text
Tensor
  ├── if CPU
  ├── if CUDA
  └── if future backend
```

With Dispatcher:

```text
Tensor
  ↓
logical op
  ↓
Dispatcher
  ├── CPU
  └── future CUDA
```

Therefore a later device/backend version can register new kernels without redesigning every Tensor method.

## 15. Reading order

```text
dispatcher.py
  ↓
kernels.py
  ↓
operator_registry.py
  ↓
tensor.py dispatched methods
  ↓
ops.py setup_context/backward
  ↓
tests/test_dispatcher.py
  ↓
examples/08_dispatcher.py
```
