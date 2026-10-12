# mini-pytorch v0.8

## 1. Version identity

- **Branch:** `v0.8`
- **Previous version:** `v0.7`
- **Theme:** Advanced Autograd Engine
- **Status:** active development
- **Target package version:** `0.8.0`
- **Starting regression baseline:** `103/103 tests passed`

v0.8 answers:

> How does an Autograd engine manage graph lifetime, repeated backward execution, retained gradients and hooks instead of merely traversing a permanent DAG?

## 2. Core difference from v0.7

v0.7 completed placement and backend routing:

```text
Tensor / Storage / Device
          ↓
       Dispatcher
      CPU / EDU
```

v0.8 deepens the execution policy above those backends:

```text
forward graph
     ↓
GradFn / Context
     ↓
Autograd Engine
 ┌────┼───────────────┐
 ▼    ▼               ▼
VJP  graph lifetime  hooks
 │    retain_graph    │
 ▼                    ▼
leaf grad       retained non-leaf grad
```

CPU/EDU placement remains orthogonal to the engine.

## 3. Planned mechanisms

### 3.1 Explicit Autograd Engine

Move backward traversal out of `Tensor.backward()` into a dedicated engine module.

Responsibilities:

- topological traversal;
- transient local gradient accumulation;
- Tensor hooks;
- leaf/non-leaf gradient materialization;
- backward-function invocation;
- saved-state release policy.

`Tensor.backward()` should become a thin public entry point.

### 3.2 Graph lifetime

`Context` gains explicit lifecycle state.

Default backward:

```text
use backward state
      ↓
release saved state
      ↓
second reuse fails clearly
```

Release must preserve Tensor values, leaves and unrelated graphs.

### 3.3 `retain_graph`

Support:

```python
loss.backward(retain_graph=True)
loss.backward()
```

The first call preserves backward state; the second may reuse it and then release it.

`retain_graph` does not reset `.grad` accumulation.

### 3.4 Non-leaf `retain_grad()`

Default:

```text
non-leaf gradient = transient engine value
Tensor.grad       = None
```

After:

```python
intermediate.retain_grad()
```

its gradient is copied/accumulated into `intermediate.grad`.

### 3.5 Tensor gradient hooks

Support:

```python
handle = tensor.register_hook(fn)
handle.remove()
```

Hooks run in registration order. A hook may:

- return `None` to observe only;
- return a replacement gradient of the same shape to transform gradient flow.

### 3.6 Custom Function workflow

Make `Function` / `Context` public teaching APIs and add a complete custom operation example proving:

- forward;
- saved tensors;
- backward VJP;
- version safety;
- graph release;
- CPU and EDU placement preservation.

### 3.7 Higher-order boundary

Do **not** claim full higher-order gradients.

Current backward formulas return NumPy arrays, so `create_graph=True` will explicitly raise `NotImplementedError` and documentation will explain what must change before true differentiable backward is possible.

## 4. Planned source changes

New:

```text
mini_pytorch/autograd_engine.py
tests/test_autograd_engine.py
docs/AUTOGRAD_ENGINE.md
examples/11_autograd_engine.py
```

Major modifications:

```text
mini_pytorch/autograd.py
mini_pytorch/tensor.py
mini_pytorch/__init__.py
pyproject.toml
README.md
```

## 5. Compatibility invariants

v0.8 must preserve:

- v0.4 saved-version/inplace safety;
- v0.5 Dispatcher Autograd wrappers;
- v0.7 CPU/EDU placement;
- device-transfer backward;
- leaf gradient accumulation;
- optimizer/data/CNN regressions.

Autograd Engine must not contain backend-specific numerical branches.

## 6. Required tests

At minimum:

- first backward succeeds;
- second backward after release fails;
- `retain_graph=True` enables reuse;
- final non-retained backward releases graph;
- shared nodes release safely once;
- non-leaf grad absent by default;
- `retain_grad()` captures correct gradient;
- retained gradients accumulate across retained executions;
- hooks observe gradients;
- hook replacement modifies upstream gradient;
- multiple hooks run in order;
- hook handle removal works;
- invalid hook gradient shape fails clearly;
- custom Function gradient correctness;
- custom Function saved-state release/version safety;
- CPU and EDU engine semantics match;
- `create_graph=True` raises explicitly;
- complete old regression stays green.

## 7. Explicit non-goals

Do not add in v0.8:

- true second derivatives / full `create_graph`;
- parallel Autograd workers;
- CUDA stream-aware engine scheduling;
- anomaly detection;
- forward-mode AD;
- DDP gradient buckets;
- compiler graph capture.

## 8. Definition of Done

Before freezing v0.8:

- [ ] explicit Engine owns backward traversal
- [ ] Context lifecycle/release implemented
- [ ] `retain_graph` implemented
- [ ] non-leaf `retain_grad()` implemented
- [ ] hooks + removable handles implemented
- [ ] custom Function teaching path documented/tested
- [ ] higher-order limitation explicit
- [ ] CPU and EDU tests pass
- [ ] all previous regression tests pass
- [ ] README/docs/example updated
- [ ] package version is `0.8.0`
- [ ] exact final v0.8 branch-head CI is green
- [ ] frozen validated tree is integrated into `main`

## 9. Next version

After v0.8, `v0.9` should focus on **Training Runtime / Serialization / Reproducibility**: unified model+optimizer+scheduler+RNG checkpoints, epoch/global-step state and deterministic resume.