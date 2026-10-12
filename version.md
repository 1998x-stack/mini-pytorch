# mini-pytorch v0.5

## 1. Version identity

- **Branch:** `v0.5`
- **Previous version:** `v0.4`
- **Theme:** operator schema + registry + DispatchKey + Dispatcher + redispatch + Autograd routing
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.5.0`
- **Validated tests:** 80/80 passed on GitHub Actions

v0.5 inserts an explicit routing layer between Tensor API and concrete operator implementations.

```text
Tensor API
   ↓
Operator Schema
   ↓
Dispatcher
   ↓
DispatchKeySet
  ┌───────┬─────────┐
  ▼       ▼         ▼
Autograd NoGrad    CPU
  │                 │
  └──── redispatch ─┘
```

The central question answered by this version is:

> Why should `x + y` resolve a logical operator through a dispatcher instead of hard-coding every backend/autograd behavior inside `Tensor.__add__`?

## 2. Core difference from v0.4

v0.4 established Tensor memory identity:

```text
Storage + shape + stride + storage_offset + version
```

v0.5 adds execution-routing identity:

```text
logical operator
      ↓
schema / registry
      ↓
dispatch keys
      ↓
selected implementation
```

This is the architectural prerequisite for future real device/backend support.

## 3. Implemented mechanisms

### Operator schema

- namespace
- operator name
- overload name
- signature metadata
- canonical qualified name
- duplicate-definition errors

### Registry

- schema registration
- `(operator, DispatchKey) -> kernel` registration
- duplicate-kernel errors
- schema/kernel introspection
- `Library` registration facade

### Dispatch keys

Educational keys:

- `Autograd`
- `NoGrad`
- `CPU`
- reserved `CUDA`

Priority:

```text
Autograd > NoGrad > CUDA > CPU
```

### Dispatcher

- key-set inference from Tensor arguments and grad mode
- dispatch priority
- kernel selection
- redispatch
- per-key fallback
- readable dispatch decisions through `explain()`
- explicit unsupported-key errors

### Tensor integration

The first dispatched Tensor operators are:

- `add`
- `mul`
- `matmul`
- `relu`
- `sum`

Other operators intentionally remain on the v0.4 direct `Function.apply()` path so the architectural delta is easy to inspect.

## 4. Autograd routing

For a requires-grad operator:

```text
Tensor op
  ↓
DispatchKeySet = {Autograd, CPU}
  ↓
Autograd wrapper
  ├── prepare backward Context
  └── redispatch without Autograd
            ↓
          {CPU}
            ↓
        CPU numerical kernel
            ↓
        attach GradFn
```

The CPU backend therefore owns numerical computation only; it does not need to know dynamic-graph policy.

## 5. NoGrad routing

Inside `no_grad()`:

```text
{NoGrad, CPU}
      ↓
NoGrad fallback
      ↓ remove NoGrad
redispatch {CPU}
      ↓
CPU kernel
```

This demonstrates that execution policy can change without duplicating backend math.

## 6. New source files

```text
mini_pytorch/dispatcher.py
mini_pytorch/kernels.py
mini_pytorch/operator_registry.py
tests/test_dispatcher.py
docs/DISPATCHER.md
examples/08_dispatcher.py
```

Important modified files:

```text
mini_pytorch/tensor.py
mini_pytorch/ops.py
mini_pytorch/__init__.py
pyproject.toml
```

## 7. Validation

Exact release-candidate head before freeze:

```text
ff84a16d1fd6d65279d0c3cb472c569c707de04b
```

GitHub Actions environment:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

Result:

```text
Ran 80 tests in 0.739s
OK
```

The suite covers:

- previous Autograd behavior
- Dataset/DataLoader/MNIST
- optimizers and LR schedulers
- Storage / stride / view / version-counter semantics
- schema/registry errors
- CPU dispatch
- Autograd priority
- NoGrad fallback
- redispatch
- dispatched gradient correctness
- inplace saved-version safety through dispatched operators

## 8. Definition of Done

- [x] operator schemas can be defined and inspected
- [x] duplicate schema registration fails clearly
- [x] kernels can be registered by DispatchKey
- [x] duplicate kernel registration fails clearly
- [x] ordinary non-grad Tensor calls route to CPU
- [x] grad-enabled calls route to Autograd before CPU
- [x] `no_grad()` selects NoGrad behavior and creates no graph
- [x] Autograd wrapper redispatches to CPU
- [x] missing operator/kernel errors are readable
- [x] selected Tensor APIs use Dispatcher
- [x] numerical and gradient behavior remains compatible with v0.4
- [x] full regression suite passes
- [x] educational dispatcher example exists
- [x] `docs/DISPATCHER.md` explains the mechanism
- [x] package version is `0.5.0`

## 9. Recommended reading order

```text
mini_pytorch/dispatcher.py
        ↓
mini_pytorch/kernels.py
        ↓
mini_pytorch/operator_registry.py
        ↓
mini_pytorch/tensor.py
        ↓
mini_pytorch/ops.py
        ↓
tests/test_dispatcher.py
        ↓
docs/DISPATCHER.md
        ↓
examples/08_dispatcher.py
```

## 10. Explicit non-goals

v0.5 intentionally does not implement:

- actual CUDA kernels
- device allocation / `.to(device)`
- production PyTorch DispatchKey complexity
- Tensor subclass dispatch
- functionalization
- Conv2d / pooling / BatchNorm
- DDP
- compiler graph capture

The CUDA key is reserved to expose the future backend boundary, not to pretend GPU support exists.

## 11. Learning goals

After v0.5, the reader should be able to explain:

1. schema vs kernel;
2. why one logical operator can have multiple implementations;
3. what a DispatchKey represents;
4. why Autograd can be a dispatch wrapper above a backend kernel;
5. why redispatch removes the current high-priority key;
6. why `no_grad()` changes routing while CPU math stays identical;
7. why Dispatcher should exist before real device/backend support.

## 12. Next version

`v0.6` should use the Dispatcher for nontrivial neural-network operators rather than adding new special cases directly to Tensor:

```text
Conv2d
MaxPool2d
Dropout
BatchNorm
CNN MNIST
```
