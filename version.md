# mini-pytorch v0.8

## 1. Version identity

- **Branch:** `v0.8`
- **Previous version:** `v0.7`
- **Theme:** Advanced Autograd Engine
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.8.0`
- **Validated regression baseline:** **123/123 tests passed**

v0.8 answers:

> How does an Autograd engine manage graph lifetime, repeated backward execution, retained gradients and hooks instead of merely traversing a permanent DAG?

## 2. Core difference from v0.7

v0.7 completed placement:

```text
Tensor / Storage / Device
          ↓
       Dispatcher
      CPU / EDU
```

v0.8 makes backward execution an explicit subsystem:

```text
forward graph
     ↓
GradFn / Context
     ↓
AutogradEngine
 ┌────┼───────────────┐
 ▼    ▼               ▼
VJP  graph lifetime  hooks
 │    retain_graph    │
 ▼                    ▼
leaf grad       retained non-leaf grad
```

Placement remains orthogonal: the Engine contains no CPU-vs-EDU execution branch.

## 3. Implemented mechanisms

### Explicit Engine

Added `mini_pytorch/autograd_engine.py`.

The Engine owns:

- topological traversal;
- transient local gradient accumulation;
- graph-availability preflight;
- Tensor hook execution;
- leaf/non-leaf persistent grad materialization;
- backward VJP invocation;
- parent-gradient shape validation;
- graph release policy.

`Tensor.backward()` is now a thin public entry point.

### Context lifecycle

`Context` now tracks `is_released` and exposes:

```text
ensure_available()
release()
```

Default successful backward releases:

- saved tensors;
- saved-version references;
- bound forward Tensor references;
- bound raw input references.

A second backward through released state fails before hook or `.grad` side effects.

### `retain_graph`

Implemented:

```python
loss.backward(retain_graph=True)
loss.backward()
```

The retained execution preserves Context state; the final non-retained execution releases the graph.

Persistent `.grad` values continue to accumulate independently of graph retention.

### Non-leaf `retain_grad()`

Implemented:

```python
intermediate.retain_grad()
```

Leaves still retain `.grad` automatically. Non-leaves remain transient by default and only persist grad after `retain_grad()`.

### Gradient hooks

Implemented:

```python
handle = tensor.register_hook(fn)
handle.remove()
```

Semantics:

- registration-order execution;
- defensive gradient copy passed to hooks;
- `None` means observe-only;
- ndarray/Tensor replacement may transform flow;
- replacement shape validation;
- Tensor replacement device validation;
- idempotent removable handles;
- transformed gradient is both accumulated and propagated upstream.

### Shared-node execution

Engine topology uses identity-based visited tracking. Contributions from multiple downstream consumers accumulate in `local_grads` before the shared node backward runs once.

### Custom Function teaching API

`Function`, `Context`, `GradFn`, `AutogradEngine` and `RemovableHandle` are public imports.

Custom Functions use the same:

```text
Function.apply → Context/GradFn → AutogradEngine
```

path as built-in educational Functions.

Tests prove custom Function:

- forward/backward correctness;
- Context release;
- saved-version safety;
- EDU device preservation.

### Higher-order boundary

`Tensor.backward(..., create_graph=True)` is explicit but raises `NotImplementedError`.

Reason: current backward formulas produce NumPy arrays rather than Tensor-valued backward operations, so a differentiable backward graph does not exist yet.

No false second-derivative claim is made.

## 4. New files

```text
mini_pytorch/autograd_engine.py
tests/test_autograd_engine.py
docs/AUTOGRAD_ENGINE.md
examples/11_autograd_engine.py
```

Major modified files:

```text
mini_pytorch/autograd.py
mini_pytorch/tensor.py
mini_pytorch/__init__.py
pyproject.toml
README.md
Agents.md
version.md
```

## 5. Preserved invariants

v0.8 preserves:

- Storage/stride/view semantics;
- shared version counter;
- inplace saved-version checks;
- Dispatcher Autograd wrappers;
- CPU/EDU placement;
- differentiable device transfer;
- leaf gradient accumulation across independent graphs;
- optimizers/schedulers;
- DataLoader/MNIST;
- CNN operator stack.

## 6. Validation

Validated release-candidate commit:

```text
f1e3f415764aa537edfe08cf7a7d0b028461ce66
```

GitHub Actions environment:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

Result:

```text
Ran 123 tests in 0.782s
OK
```

This is the previous v0.7 baseline of 103 tests plus 20 dedicated v0.8 tests.

## 7. v0.8-specific test coverage

- default graph release;
- repeated backward error;
- retain_graph reuse and final release;
- shared-node accumulation/release;
- CPU/EDU lifetime parity;
- create_graph explicit refusal without graph consumption;
- default non-leaf `.grad` absence;
- retain_grad materialization;
- retained-grad accumulation;
- retain_grad on EDU;
- hook observe/transform;
- hook ordering;
- transformed hook + retained grad;
- removable handle idempotence;
- invalid hook shape;
- hook Tensor device mismatch;
- custom Function forward/backward;
- custom Function version safety;
- custom Function EDU placement;
- Tensor gradient-seed device mismatch.

## 8. Definition of Done

- [x] explicit Engine owns backward traversal
- [x] Context lifecycle/release implemented
- [x] `retain_graph` implemented
- [x] non-leaf `retain_grad()` implemented
- [x] hooks + removable handles implemented
- [x] custom Function teaching path documented/tested
- [x] higher-order limitation explicit
- [x] CPU and EDU semantics match
- [x] all previous regression tests pass
- [x] README/docs/example updated
- [x] package version is `0.8.0`
- [x] release candidate CI is green
- [x] branch marked frozen pending exact frozen-head CI/integration

## 9. Recommended reading order

```text
mini_pytorch/autograd.py
        ↓
mini_pytorch/autograd_engine.py
        ↓
mini_pytorch/tensor.py
        ↓
tests/test_autograd_engine.py
        ↓
docs/AUTOGRAD_ENGINE.md
        ↓
examples/11_autograd_engine.py
```

## 10. Explicit non-goals

v0.8 intentionally does not implement:

- true higher-order gradients;
- full `create_graph=True`;
- forward-mode AD;
- parallel Autograd workers;
- anomaly detection;
- stream-aware CUDA scheduling;
- DDP gradient buckets;
- compiler graph capture.

## 11. Learning goals

After v0.8 the reader should be able to explain:

1. graph structure vs backward saved state vs Engine-transient gradients;
2. why default backward releases state;
3. why `retain_graph`, `retain_grad` and `zero_grad` are different mechanisms;
4. why shared graph nodes need local gradient accumulation;
5. where hooks sit in gradient flow;
6. why non-leaf `.grad` is absent by default;
7. how custom Function uses the same Engine path;
8. why NumPy-valued backward formulas prevent real higher-order Autograd.

## 12. Next version

`v0.9` should focus on **Training Runtime / Serialization / Reproducibility**:

```text
model state
optimizer state
scheduler state
RNG state
epoch/global-step state
      ↓
unified checkpoint
      ↓
deterministic resume
```
