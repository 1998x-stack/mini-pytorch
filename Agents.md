# Agents.md — mini-pytorch v0.12 Frozen Branch Contract

> Read this file before modifying frozen branch `v0.12`.

## Branch identity

- Branch: `v0.12`
- Previous: `v0.11`
- Theme: Guards / Graph Cache / Recompile
- Status: **frozen historical learning snapshot**
- Package: `0.12.0`
- Validated baseline: **192/192 tests passed**

## Historical promise

v0.12 exists to teach:

```text
Graph IR
   ↓
runtime InputSignature
   ↓
inspectable GuardSet
   ↓
GraphCache
 ┌────┴────┐
 HIT       MISS
  │          │
 replay    capture
```

One callable may own multiple guarded Graph specializations.

## Frozen specialization model

The v0.12 contract guards exactly:

```text
input count
shape
dtype
device
requires_grad
grad mode
```

A guard failure is a normal cache miss and may create another specialization.

Do not retrofit symbolic shapes or production Dynamo semantics into this branch.

## Frozen compiler invariants

- capture remains at outer `Dispatcher.dispatch` logical-op level;
- redispatch/backend execution is not duplicated into IR nodes;
- GraphInterpreter replays through Dispatcher;
- cache hits never call backend kernels directly;
- incompatible entries are not mutated into new shapes/devices;
- new specializations append separate cache entries;
- guard decisions remain inspectable;
- `explain()` must not mutate cache statistics;
- v0.11 GraphBreak behavior remains explicit;
- GraphBreak installs no cache entry;
- closure Tensor constants remain capture-by-value snapshots;
- CPU and EDU remain real distinct device identities even though both use NumPy-hosted educational storage.

## Allowed future changes on this frozen branch

Only:

- documentation corrections;
- tests clarifying already-promised v0.12 behavior;
- narrowly scoped bug fixes that preserve v0.12 semantics.

Do not backport:

- AOTAutograd-like graph differentiation;
- symbolic shapes;
- Python frame interception;
- lowering/fusion;
- generated kernels;
- future compiler runtime features.

## Required regression command

```bash
python -m unittest discover -s tests -v
```

Historical validated result:

```text
Ran 192 tests
OK
```

## Key files

```text
mini_pytorch/compiler/signature.py
mini_pytorch/compiler/guards.py
mini_pytorch/compiler/cache.py
mini_pytorch/compiler/compiled.py
mini_pytorch/compiler/capture.py
mini_pytorch/compiler/interpreter.py
mini_pytorch/compiler/ir.py

tests/test_graph_cache.py
tests/test_graph_ir.py
docs/GRAPH_CACHE.md
docs/GRAPH_IR.md
examples/15_graph_cache.py
```

## Reading order

```text
version.md
   ↓
signature.py
   ↓
guards.py
   ↓
cache.py
   ↓
compiled.py
   ↓
test_graph_cache.py
   ↓
GRAPH_CACHE.md
```

## Next-version boundary

The next architectural problem belongs in `v0.13`, created from latest `main` after v0.12 integration:

> Once a guarded forward Graph exists, how can backward computation itself become an explicit graph representation suitable for ahead-of-time transformation?
