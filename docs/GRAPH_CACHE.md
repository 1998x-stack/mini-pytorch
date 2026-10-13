# Guards, Graph Cache and Recompile — v0.12

## 1. The new correctness question

v0.11 proved that one eager execution can be represented as Graph IR and replayed.

That is not enough for a compiled wrapper.

A captured graph observed concrete assumptions such as:

```text
input[0].shape == (2, 4)
input[0].dtype == float64
input[0].device == cpu
input[0].requires_grad == True
grad_enabled == True
```

A future call may not satisfy those assumptions.

v0.12 therefore separates:

```text
Graph
= logical program captured in v0.11

Guards
= assumptions under which one specialization is reusable

GraphCache
= collection of guarded specializations for one callable
```

## 2. Runtime signature

`TensorSignature` is deliberately value-independent:

```text
shape
dtype
device
requires_grad
```

`InputSignature` adds call policy:

```text
tuple[TensorSignature, ...]
grad_enabled
```

Tensor element values do **not** participate in the input signature.

Two tensors with equal metadata but different numerical values should hit the same Graph.

## 3. Why specialize on grad mode?

v0.11 already showed that IR stores logical operators and replay chooses Dispatcher policy dynamically.

So it would be technically possible to reuse one Graph under both Autograd and NoGrad.

v0.12 intentionally chooses a more conservative educational policy:

```text
grad enabled
!=
grad disabled
```

This makes execution-context assumptions visible and creates an explicit recompile example.

This is an educational policy, not a claim that every real compiler uses exactly this guard set.

## 4. GuardSet

A cache entry owns an ordered `GuardSet`.

For one Tensor input it looks like:

```text
input_count == 1
input[0].shape == (2, 4)
input[0].dtype == 'float64'
input[0].device == 'cpu'
input[0].requires_grad == True
grad_enabled == True
```

Each guard returns an inspectable `GuardResult`:

```text
passed
label
expected
actual
reason
```

A failed guard is normal information:

```text
guard FAIL
    ↓
cache entry does not match
    ↓
try next entry
```

It is not an exception.

## 5. One callable, many graphs

```text
CompiledFunction(fn)
        │
        ▼
     GraphCache
   ┌────┼─────────┐
   ▼    ▼         ▼
entry0 entry1   entry2
 S1     S2       S3
```

The first educational cache uses deterministic linear search.

That is intentionally simpler than hashing because the goal is to see every guard decision.

## 6. Cache hit

```text
runtime inputs
      ↓
InputSignature
      ↓
entry0 GuardSet PASS
      ↓
entry0.hit_count += 1
      ↓
GraphInterpreter(entry0.graph)
      ↓
Dispatcher
      ↓
current Autograd / backend policy
```

Notice that a cache hit does **not** jump directly to NumPy kernels.

The v0.11 architecture remains intact.

## 7. Cache miss and compile

```text
no GuardSet passes
      ↓
miss_count += 1
      ↓
capture(fn, *runtime_inputs)
      ↓
Graph + eager_output
      ↓
new GuardSet
      ↓
new GraphCacheEntry
```

The miss path returns `captured.eager_output` because capture already executed the function.

Immediately replaying the new Graph would execute the program twice.

## 8. Recompile means new specialization

Example:

```text
A1 shape (2, 4)
MISS -> compile_id 0

A2 shape (2, 4)
HIT  -> compile_id 0

B1 shape (8, 4)
MISS -> compile_id 1

A3 shape (2, 4)
HIT  -> compile_id 0
```

Then:

```text
compile_count   = 2
recompile_count = 1
cache_entries   = 2
hits            = 2
misses          = 2
```

A recompile does not mutate graph0 into graph1.

Both specializations stay available.

## 9. `explain()`

Guard systems become difficult to learn when cache decisions are opaque.

`compiled.explain(*inputs)` checks all entries without modifying counters.

Example conceptually:

```text
entry 0: FAIL
  FAIL input[0].shape: expected (2, 4), actual (8, 4)

entry 1: PASS
  all guards passed
```

This provides a small analogue of compiler guard-debugging tools.

## 10. GraphBreak interaction

GraphBreak remains the v0.11 boundary for unsupported eager dataflow.

If capture raises `GraphBreakError`:

```text
cache miss happened
      ↓
capture attempted
      ↓
GraphBreakError
      ↓
NO cache entry
NO successful compile increment
graph_break_count += 1
```

v0.12 deliberately propagates the error.

It does not automatically rerun the Python function, because a failed trace may already have executed side effects before the break.

## 11. Constant policy

Tensor constants discovered outside graph inputs are copied into `TensorConstant` values by v0.11.

v0.12 keeps that rule:

```text
compile-time closure Tensor
      ↓
copy value into Graph
      ↓
cache hit replays frozen value
```

Therefore a mutable value that should participate in specialization must be an explicit graph input in this version.

General Python closure guarding belongs to a more advanced Dynamo-like system.

## 12. Why no symbolic shapes yet?

The simple rule:

```text
shape changed -> guard fail -> new specialization
```

is easy to inspect and correct.

Symbolic shapes require a different model:

```text
shape = (N, 4)
constraints: N > 0
specialization validity based on symbolic predicates
```

That is valuable, but it is not necessary to understand the first guard/cache/recompile loop.

## 13. Connection to future versions

```text
v0.11
Graph IR
   ↓
v0.12
Guards + Graph Cache
   ↓
v0.13
AOTAutograd-like graph differentiation
   ↓
v0.14
Lowering / fusion
```

v0.12 creates a stable notion of **which graph specialization is valid** before later compiler stages transform that graph.
