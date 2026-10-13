# Agents.md — mini-pytorch v0.12 Development Contract

> Read before modifying branch `v0.12`.

## Branch identity

- Branch: `v0.12`
- Previous: `v0.11`
- Theme: Guards / Graph Cache / Recompile
- Status: active until exact final-head CI passes

## Mandatory reading order

1. `Agents.md`
2. `version.md`
3. `docs/GRAPH_IR.md`
4. `tests/test_graph_ir.py`
5. `mini_pytorch/compiler/*`
6. relevant Dispatcher/Autograd tests

## Allowed scope

v0.12 may add:

```text
TensorSignature
InputSignature
Guard / GuardResult
GuardSet
GraphCacheEntry
GraphCache
CompiledFunction
compile_graph(fn)
cache/recompile statistics
explain surface
v0.12 tests/docs/example
```

## Required architecture

```text
runtime call
    ↓
InputSignature
    ↓
GuardSet
    ↓
GraphCache
 ┌──────┴──────┐
 ▼             ▼
hit           miss
 │              │
GraphInterpreter capture
 │              ↓
 │          new entry
 └──────────────┘
```

Rules:

- guard logic stays outside numerical kernels;
- Graph IR stays logical and does not store DispatchKey decisions;
- cache hit replays with `GraphInterpreter` through normal Dispatcher;
- cache miss captures a new Graph; do not mutate an incompatible Graph;
- guard failures are normal cache-miss information, not exceptional control flow;
- cache decisions and failure reasons must be inspectable;
- exact specialization fields for v0.12: input count, shape, dtype, device, requires_grad, grad mode;
- no symbolic-shape claims;
- v0.11 GraphBreak semantics remain explicit;
- closure Tensor constants remain capture-by-value snapshots in this version.

## GraphBreak policy

If capture fails with `GraphBreakError`:

```text
no cache entry is installed
successful compile_count does not increase
graph_break_count increases
error propagates to caller
```

Do not automatically re-run the Python function after a failed trace because that could duplicate side effects.

## Historical invariants

All earlier tests must remain green, especially:

- v0.4 Storage/view/version safety;
- v0.5 Dispatcher/redispatch;
- v0.7 device/backend routing;
- v0.8 Autograd Engine;
- v0.9 deterministic resume;
- v0.10 DDP equivalence;
- v0.11 eager-vs-Graph replay forward/backward equivalence.

## Definition of Done

Before freezing v0.12:

- full regression green;
- multi-specialization cache proof green;
- grad-mode specialization tested;
- CPU/EDU specialization tested;
- eager/compiled output and gradient equivalence tested;
- explain/stats tested;
- GraphBreak/cache interaction tested;
- README/version/docs/example/manifest updated;
- exact final branch-head GitHub Actions green;
- then freeze branch and integrate validated tree into main.
