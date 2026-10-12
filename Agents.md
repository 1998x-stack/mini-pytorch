# Agents.md — mini-pytorch v0.8 Frozen Branch Contract

> Read this file before modifying the frozen `v0.8` branch.

## Branch identity

- Branch: `v0.8`
- Previous version: `v0.7`
- Primary mental model: **Advanced Autograd Engine**
- Status: **frozen historical learning snapshot**
- Package version: `0.8.0`
- Validated baseline: `123/123 tests passed`

## 1. What v0.8 teaches

```text
GradFn / Context
      ↓
AutogradEngine
 ┌────┼──────────────┐
 ▼    ▼              ▼
VJP  graph lifetime hooks
      retain_graph    │
                      ▼
              retained gradients
```

Frozen v0.8 includes:

- explicit backward Engine;
- Context release state;
- repeated-backward failure after release;
- `retain_graph`;
- non-leaf `retain_grad()`;
- gradient hooks;
- removable hook handles;
- public custom `Function` workflow;
- explicit `create_graph=True` non-support boundary;
- CPU/EDU parity for Autograd runtime semantics.

## 2. Historical branch rule

This branch is frozen.

Allowed changes only:

- documentation corrections;
- tests clarifying promised v0.8 behavior;
- narrowly scoped correctness fixes inside v0.8 Autograd semantics.

Forbidden:

- v0.9 training-runtime/checkpoint work;
- DDP / ProcessGroup / AllReduce;
- real CUDA runtime/streams;
- compiler capture;
- forward-mode AD;
- unrelated NN/operator expansion.

## 3. Frozen graph-lifetime semantics

Default:

```text
successful backward
      ↓
release all reachable Context backward state
      ↓
reusing graph fails before hook/.grad side effects
```

Retained execution:

```python
loss.backward(retain_graph=True)
loss.backward()
```

The first preserves backward state. The final non-retained execution releases it.

Release happens only after the whole backward execution succeeds, preventing half-released graphs after hook/backward exceptions.

## 4. Frozen gradient-storage semantics

```text
leaf Tensor
→ persistent .grad accumulation

non-leaf Tensor
→ Engine-local transient grad by default

non-leaf + retain_grad()
→ persistent .grad accumulation too
```

`retain_graph`, `retain_grad`, and `zero_grad` are separate mechanisms and must not be conflated.

## 5. Frozen hook semantics

`register_hook(fn)`:

- requires `requires_grad=True`;
- runs in registration order;
- receives a defensive gradient copy;
- `None` means observe-only;
- replacement gradient must match shape;
- Tensor replacement must match device;
- transformed gradient is both persisted (when applicable) and propagated;
- `RemovableHandle.remove()` is idempotent.

## 6. Custom Function semantics

Custom Functions use the same path as built-ins:

```text
Function.apply
   ↓
Context + GradFn
   ↓
AutogradEngine
```

Do not add a special execution path for custom Functions.

Saved-version checks and graph-lifetime rules apply equally to them.

## 7. Higher-order boundary

The current Engine propagates NumPy gradient arrays.

Therefore true higher-order Autograd is not implemented.

```python
loss.backward(create_graph=True)
```

must continue to raise a clear `NotImplementedError` unless a later version redesigns backward formulas to operate on Tensor objects and construct a differentiable backward graph.

## 8. Device/backend invariant

The Engine is placement-agnostic.

Do not add:

```python
if tensor.device == "cpu": ...
elif tensor.device == "edu": ...
```

inside Autograd execution policy.

CPU/EDU placement remains owned by Tensor/Storage/Dispatcher boundaries established in v0.7.

## 9. Validation baseline

Release candidate:

```text
f1e3f415764aa537edfe08cf7a7d0b028461ce66
```

GitHub Actions:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
123 tests
OK
```

Before reporting any future narrowly scoped v0.8 fix as validated, rerun the complete suite and require GitHub Actions success on the exact modified branch head.

## 10. Documentation map

```text
README.md
version.md
mini_pytorch/autograd.py
mini_pytorch/autograd_engine.py
mini_pytorch/tensor.py
tests/test_autograd_engine.py
docs/AUTOGRAD_ENGINE.md
examples/11_autograd_engine.py
```

## 11. Next-version boundary

Training Runtime / Serialization / Reproducibility work belongs to `v0.9`, created from the latest validated `main`, not from this frozen branch.