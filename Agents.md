# Agents.md — mini-pytorch v0.5 Development Contract

> Read this file before modifying the `v0.5` branch.

## Branch identity

- Branch: `v0.5`
- Previous version: `v0.4`
- Primary mental model: **operator schema + registry + DispatchKey + Dispatcher + redispatch + Autograd routing**
- Status: active until Definition of Done is satisfied, then freeze this branch.

## 1. Scope rule

v0.5 should teach one architectural idea:

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

Do not turn v0.5 into a general feature release.

Allowed core work:

- operator schema
- operator/kernel registry
- DispatchKey and DispatchKeySet
- Dispatcher
- dispatch priority
- redispatch
- fallback
- CPU reference kernels for selected operators
- Autograd wrapper routing
- NoGrad routing
- selected Tensor integration: add, mul, matmul, relu, sum
- dispatcher tests/docs/example

## 2. Explicit non-goals

Do not add in this branch:

- real CUDA backend
- device allocation / `.to()`
- Conv2d / pooling / BatchNorm
- multiprocessing DataLoader
- DDP
- advanced autograd graph lifetime
- compiler graph capture
- Tensor subclass dispatch
- production-scale PyTorch DispatchKey taxonomy

`CUDA` may exist only as a reserved educational key.

## 3. Preserve v0.4 invariants

The following must remain correct:

- explicit Storage
- shape / stride / storage_offset
- view aliasing
- contiguous semantics
- version counter
- inplace saved-tensor safety
- scalar Tensor shape `()`
- existing Autograd numerical behavior
- optimizer and DataLoader behavior

Dispatcher work must not silently weaken the memory model.

## 4. Dependency direction

Preferred structure:

```text
Tensor
  ↓
Dispatcher
  ↓
operator registry
  ↓
CPU kernels

Dispatcher
  ↓
Autograd wrapper
  ↓ redispatch
CPU kernels
```

Avoid importing `nn`, `optim` or `data` into Dispatcher/core operator files.

Avoid a Tensor ↔ Dispatcher eager import cycle; Tensor operator methods should use local imports when necessary.

## 5. Backend vs Autograd rule

CPU kernels must own only numerical behavior.

Bad:

```text
CPU add kernel
├── NumPy add
└── builds GradFn
```

Desired:

```text
Autograd dispatch wrapper
   ↓
prepare Context
   ↓
redispatch
   ↓
CPU numerical kernel
   ↓
attach GradFn
```

The educational point is that backend kernels should not need to know dynamic-graph policy.

## 6. Redispatch rule

A wrapper selected at one DispatchKey must be able to invoke the same logical operator with that key removed.

```text
Autograd + CPU
      ↓ choose Autograd
Autograd wrapper
      ↓ remove Autograd
CPU
      ↓
CPU kernel
```

Do not implement redispatch by calling the public Tensor operator again; that would recurse into the same high-priority key.

## 7. Registration rules

Tests must cover:

- duplicate schema error
- kernel registration for an undefined operator error
- duplicate `(operator, DispatchKey)` kernel error
- unknown operator error
- missing kernel/fallback error
- introspection of registered schema/kernels

## 8. Testing requirements

Run the full suite:

```bash
python -m unittest discover -s tests -v
```

v0.5-specific tests must include:

- schema/registry unit tests
- dispatch priority
- CPU routing
- NoGrad fallback and redispatch
- Autograd-before-CPU routing
- gradient correctness through dispatched operators
- selected Tensor API integration
- readable error paths

Do not claim v0.5 complete until GitHub Actions passes on the exact final branch head.

## 9. Documentation requirements

Before freeze, update:

- `version.md`
- `README.md`
- `docs/DISPATCHER.md`
- `examples/08_dispatcher.py`

The documentation must explicitly answer:

1. schema vs kernel — what is the difference?
2. DispatchKey — what does it represent?
3. dispatch priority — why does Autograd beat CPU?
4. redispatch — why is it needed?
5. NoGrad — why can routing change while CPU math stays identical?
6. why Dispatcher comes before real device/backend support.

## 10. Freeze / integration

When complete:

1. full CI green on exact `v0.5` head;
2. mark `version.md` frozen;
3. freeze branch as historical snapshot;
4. synchronize validated implementation into `main`;
5. restore `main` canonical `Agents.md` and advance `Plan.md` to v0.6;
6. verify `main` is not behind `v0.5`.

After freeze, only documentation corrections or narrowly scoped v0.5 bug fixes are allowed.