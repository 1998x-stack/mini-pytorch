# mini-pytorch v0.5

## 1. Version identity

- **Branch:** `v0.5`
- **Previous version:** `v0.4`
- **Theme:** operator schema + registry + DispatchKey + Dispatcher + redispatch + Autograd routing
- **Status:** active development
- **Target package version:** `0.5.0`

v0.5 introduces the routing layer between Tensor API and concrete operator implementations.

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

The primary question is:

> Why should `x + y` resolve an operator through a dispatcher instead of hard-coding every backend/autograd behavior inside `Tensor.__add__`?

## 2. Difference from v0.4

v0.4 established Tensor memory identity:

```text
Storage + shape + stride + offset + version
```

v0.5 keeps those semantics stable and adds an execution-routing identity:

```text
logical operator
      ↓
schema / registry
      ↓
dispatch keys
      ↓
selected implementation
```

This is the prerequisite for future device/backend work because Tensor methods no longer need large `if device == ...` branches.

## 3. Required mechanisms

### Operator schema

- namespace
- operator name
- overload name
- signature metadata
- canonical qualified name
- duplicate-definition errors

### Registry

- schema registration
- per-operator per-key kernel registration
- duplicate-kernel errors
- lookup/introspection API

### Dispatch keys

Educational keys:

- `Autograd`
- `NoGrad`
- `CPU`
- reserved `CUDA`

### Dispatcher

- key-set inference from Tensor arguments and grad mode
- dispatch priority
- operator lookup
- kernel selection
- redispatch
- fallback behavior
- readable dispatch explanation/introspection

### Tensor integration

The first dispatched Tensor operators are:

- add
- mul
- matmul
- relu
- sum

Other operators remain on the previous direct `Function.apply()` path so the version delta stays easy to inspect.

### Autograd routing

For dispatched differentiable operators:

```text
Tensor op
  ↓
Autograd key selected
  ↓
Autograd wrapper prepares backward Context
  ↓
redispatch without Autograd
  ↓
CPU kernel
  ↓
attach GradFn to result
```

The CPU backend therefore does not need to own dynamic-graph logic.

## 4. Architecture boundaries

```text
Tensor
  │ calls logical op
  ▼
Dispatcher
  │
  ├── Autograd wrapper
  │      │
  │      └── redispatch
  │              ↓
  ├── NoGrad fallback
  │      │
  │      └── redispatch
  │              ↓
  └── CPU kernel
```

Dependency direction:

```text
Tensor → Dispatcher → registered kernels
                  ↘ Autograd wrappers
```

Core backend kernels should not depend on Module/Optimizer/DataLoader.

## 5. Acceptance criteria

v0.5 is complete when tests demonstrate:

- [ ] operator schemas can be defined and inspected
- [ ] duplicate schema registration fails clearly
- [ ] kernels can be registered by DispatchKey
- [ ] duplicate kernel registration fails clearly
- [ ] ordinary non-grad Tensor calls route to CPU
- [ ] grad-enabled calls route to Autograd before CPU
- [ ] `no_grad()` selects NoGrad behavior and creates no graph
- [ ] Autograd wrapper redispatches to CPU
- [ ] missing operator/kernel errors are readable
- [ ] selected Tensor APIs use Dispatcher
- [ ] numerical and gradient behavior remains compatible with v0.4
- [ ] full regression suite passes
- [ ] educational dispatcher example runs
- [ ] `docs/DISPATCHER.md` explains the mechanism

## 6. Explicit non-goals

v0.5 does not implement:

- actual CUDA kernels
- device allocation or `.to(device)`
- production PyTorch DispatchKey complexity
- Tensor subclass dispatch
- functionalization
- composite implicit/explicit autograd key families
- Conv2d
- DDP
- compiler graph capture

The `CUDA` key is reserved to make the future backend boundary visible, not to pretend GPU support already exists.

## 7. Recommended reading order

Once implemented:

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

## 8. Learning goals

By the end of v0.5, a reader should be able to explain:

1. operator schema and kernel implementation are different concepts;
2. why one operator can have multiple implementations;
3. what a DispatchKey represents;
4. why Autograd can be a routing layer above a backend kernel;
5. what redispatch means and why it prevents recursive dispatch;
6. why `no_grad()` can affect routing without changing the CPU math kernel;
7. why a Dispatcher should exist before adding real device backends.

## 9. Next version direction

After v0.5 is frozen, `v0.6` can use the Dispatcher for nontrivial neural-network operators such as Conv2d, pooling, Dropout and BatchNorm rather than adding more special cases directly to Tensor.