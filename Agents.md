# Agents.md — mini-pytorch v0.8 Branch Contract

> Mandatory reading before changing the `v0.8` branch.

## Branch identity

- Branch: `v0.8`
- Previous version: `v0.7`
- Primary mental model: **Advanced Autograd Engine**
- Status: active development

## 1. Allowed scope

v0.8 may implement only the Autograd-runtime concepts needed to make dynamic backward execution explicit:

```text
GradFn / Context
      ↓
Autograd Engine
 ┌────┼──────────────┐
 ▼    ▼              ▼
VJP  graph lifetime hooks
      retain_graph    │
                      ▼
              retained gradients
```

Allowed:

- graph traversal refactor into a dedicated engine;
- Context release/lifetime state;
- `retain_graph`;
- non-leaf `retain_grad()`;
- gradient hooks and removable handles;
- public custom `Function` teaching workflow;
- explicit higher-order-gradient boundary;
- tests/docs/examples required to explain those mechanisms.

## 2. Forbidden scope

Do not add:

- DDP / ProcessGroup / AllReduce;
- true CUDA runtime or streams;
- new accelerator backend;
- compiler capture;
- serialization/runtime checkpoint system;
- forward-mode AD;
- fake/full `create_graph=True` support without differentiable backward formulas;
- unrelated NN/operator expansion.

Those belong to later versions.

## 3. Mandatory invariants

Preserve all established architecture:

- Storage/view/version-counter safety;
- saved-version checks for inplace mutation;
- Dispatcher Autograd wrappers;
- CPU and EDU placement semantics;
- device-transfer backward;
- Module/Optimizer/DataLoader/CNN behavior;
- leaf `.grad` accumulation across independent graphs.

Autograd Engine must not branch on `cpu` vs `edu`.

## 4. Engine design rules

`Tensor.backward()` should remain user-facing but should not own the traversal implementation.

Prefer:

```text
Tensor.backward(...)
      ↓
autograd_engine.engine.backward(...)
      ↓
traverse graph
      ↓
apply hooks
      ↓
materialize requested grads
      ↓
call backward VJPs
      ↓
release or retain contexts
```

The Engine owns transient local gradients; Tensor owns only persistent user-observable `.grad`.

## 5. Graph lifetime semantics

Default backward releases traversed backward state after each node has executed successfully.

A released Context must fail clearly if reused.

`retain_graph=True` preserves graph state for another backward execution.

Do not confuse graph retention with gradient reset:

```text
retain_graph controls backward-state lifetime
zero_grad controls persistent grad accumulation
```

## 6. `retain_grad` semantics

- leaves already retain/accumulate `.grad`;
- non-leaves do not retain grad by default;
- `retain_grad()` marks a requires-grad non-leaf for persistent gradient capture;
- retained grad should accumulate across repeated backward executions just like leaf grad;
- detached/non-requires-grad Tensor should reject `retain_grad()` clearly.

## 7. Hook semantics

`register_hook(fn)`:

- requires `requires_grad=True`;
- executes hooks in registration order;
- hook receives a defensive gradient array copy;
- `None` means no replacement;
- replacement must preserve shape;
- replacement is converted to the Tensor dtype;
- transformed gradient is the one accumulated and propagated;
- returned handle supports idempotent `remove()`.

## 8. Custom Function rule

Do not special-case custom functions in the Engine. They must use the same `Function.apply → GradFn → Context → Engine` path as built-in educational Functions.

The example must demonstrate saved tensor state and version safety.

## 9. Higher-order rule

The current engine uses NumPy arrays as backward values.

Therefore full higher-order gradients are not supported. If a `create_graph` argument is exposed, it must raise a clear `NotImplementedError` until backward formulas themselves operate on Tensor objects and build a new graph.

## 10. Testing contract

Starting baseline:

```text
v0.7
103 tests
OK
```

Add dedicated v0.8 coverage for:

- release/reuse errors;
- retain_graph reuse;
- shared graph nodes;
- retain_grad;
- hook order/replacement/removal/errors;
- custom Function;
- CPU/EDU parity;
- `create_graph` boundary.

Run full suite:

```bash
python -m unittest discover -s tests -v
```

Do not freeze unless GitHub Actions passes on the exact final v0.8 branch HEAD.

## 11. Documentation contract

Before freeze update:

```text
README.md
version.md
Agents.md
docs/AUTOGRAD_ENGINE.md
examples/11_autograd_engine.py
pyproject.toml
```

## 12. Freeze rule

When the exact final branch head is green:

1. mark `version.md` frozen;
2. mark this file frozen;
3. do not change framework code again;
4. fast-forward validated v0.8 into main;
5. restore canonical main README/Agents/Plan;
6. move roadmap to v0.9 Training Runtime / Reproducibility.