# mini-pytorch v0.12

## 1. Version identity

- **Branch:** `v0.12`
- **Previous version:** `v0.11`
- **Theme:** Guards / Graph Cache / Recompile
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.12.0`
- **Validated regression baseline:** **192/192 tests passed**

v0.12 answers:

> Once eager execution has been captured as Graph IR, what assumptions make that graph valid for a future call, and how should the runtime choose between cache hit and recapture?

## 2. Core difference from v0.11

v0.11 established:

```text
one eager execution
      ↓
Graph IR
      ↓
GraphInterpreter replay
```

v0.12 adds specialization and reuse:

```text
runtime Tensor inputs
      ↓
InputSignature
      ↓
GuardSet
      ↓
GraphCache
  ┌────┴────┐
 HIT       MISS
  │          │
 replay    capture
             ↓
          new entry
```

One callable may own multiple guarded Graph specializations.

## 3. Implemented components

### Runtime signatures

```text
TensorSignature
├── shape
├── dtype
├── device
└── requires_grad

InputSignature
├── tensors
└── grad_enabled
```

### Guards

```text
InputCountGuard
TensorShapeGuard
TensorDTypeGuard
TensorDeviceGuard
RequiresGradGuard
GradModeGuard
GuardResult
GuardSetResult
GuardSet
```

Every guard decision is inspectable with expected/actual values and a reason.

### Graph cache

```text
GraphCacheEntry
EntryCheck
CacheLookup
GraphCache
```

The cache uses deterministic linear search so each guard decision remains visible.

### Compiled wrapper

```python
compiled = compile_graph(fn)
compiled(*inputs)
compiled.stats()
compiled.explain(*inputs)
compiled.clear_cache()
```

A miss returns the eager output already produced by capture; it does not replay immediately and execute the program twice.

## 4. Specialization policy

v0.12 deliberately uses conservative exact specialization:

```text
input count
shape
dtype
device
requires_grad
grad mode
```

A mismatch means:

```text
guard failure
      ↓
normal cache miss
      ↓
new specialization
```

It is not a runtime error.

## 5. Recompile semantics

Validated sequence:

```text
A1 shape S1 → MISS → compile graph0
A2 shape S1 → HIT graph0
B1 shape S2 → MISS → compile graph1
A3 shape S1 → HIT graph0
```

Expected and tested:

```text
compile_count   = 2
recompile_count = 1
cache_entries   = 2
hit_count       = 2
miss_count      = 2
```

The old entry remains reusable after a different specialization is compiled.

## 6. Execution-policy specialization

Grad mode participates in the signature:

```text
same Tensor metadata
+
grad enabled
!=
grad disabled
```

This is an explicit educational policy. Graph IR itself still stores logical operators rather than Autograd/NoGrad DispatchKeys.

## 7. Backend specialization

CPU and EDU calls with identical shapes/dtypes receive separate cache entries because `device` is guarded.

A later CPU call correctly returns to the original CPU specialization.

## 8. Forward/backward correctness

Compiled cache hits replay through:

```text
GraphInterpreter
      ↓
Dispatcher
      ↓
current Autograd/backend routing
```

Tests prove compiled outputs and input gradients match eager reference execution.

## 9. Explainability

`compiled.explain(*inputs)` evaluates every cache entry without mutating counters.

It reports information such as:

```text
entry 0: FAIL
  input[0].shape expected (2, 3), actual (8, 3)

entry 1: PASS
```

This keeps guard/recompile behavior visible instead of magical.

## 10. GraphBreak policy

If v0.11 capture hits unsupported dynamic dataflow:

```text
cache miss
    ↓
capture attempt
    ↓
GraphBreakError
    ↓
NO cache entry
compile_count unchanged
graph_break_count += 1
```

v0.12 deliberately propagates the GraphBreak instead of automatically re-running the Python function.

## 11. Constant policy

Tensor constants discovered outside explicit graph inputs remain captured by value:

```text
closure Tensor at compile time
      ↓
TensorConstant snapshot
      ↓
cache hit replays frozen value
```

Values that must vary should be explicit Tensor graph inputs.

General closure-cell guards are not implemented.

## 12. Important files

```text
mini_pytorch/compiler/signature.py
mini_pytorch/compiler/guards.py
mini_pytorch/compiler/cache.py
mini_pytorch/compiler/compiled.py

tests/test_graph_cache.py
docs/GRAPH_CACHE.md
examples/15_graph_cache.py
```

Recommended reading order:

```text
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
    ↓
15_graph_cache.py
```

## 13. Tests added in v0.12

The regression suite grew from 176 to 192 tests.

New coverage includes:

- signature equality/hash;
- shape/dtype/device/requires-grad specialization;
- grad-mode specialization;
- inspectable guard failures;
- first-call miss / second-call hit;
- multi-specialization cache reuse;
- multi-input specialization;
- CPU/EDU separate entries;
- eager-vs-compiled forward equality;
- eager-vs-compiled input-gradient equality;
- non-mutating explain();
- GraphBreak/cache interaction;
- capture-by-value closure constant policy;
- cache clearing and counter reset.

## 14. Non-goals

v0.12 does not implement:

- Python frame/bytecode interception;
- symbolic/dynamic shapes;
- general Python argument specialization;
- production Dynamo guards;
- automatic graph partitioning;
- AOTAutograd;
- lowering/fusion/code generation.

## 15. Next version

v0.13 should study **AOTAutograd-like graph differentiation**: making forward and backward graph structure explicit instead of leaving backward entirely inside the eager AutogradEngine.
