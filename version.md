# mini-pytorch v0.12

## 1. Version identity

- **Branch:** `v0.12`
- **Previous version:** `v0.11`
- **Theme:** Guards / Graph Cache / Recompile
- **Status:** active development
- **Package version:** `0.12.0`
- **Starting regression baseline:** **176/176 tests passed**

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

One Python callable may therefore own multiple guarded graph specializations.

## 3. Specialization policy

The first educational policy is intentionally conservative and exact:

```text
input count
shape
dtype
device
requires_grad
grad mode
```

A difference in any guarded field is a cache miss and may create a new specialization.

This version does **not** claim symbolic/dynamic shape support.

## 4. Constants policy

Tensor constants discovered during v0.11 capture remain captured **by value**.

```text
closure Tensor during successful compile
      ↓
TensorConstant snapshot in Graph
      ↓
future cache hit replays snapshot
```

v0.12 does not inspect Python closure cells or guard arbitrary Python object mutation. Values that must vary between calls should be explicit Tensor graph inputs.

## 5. Planned public APIs

```python
TensorSignature.from_tensor(x)
InputSignature.from_inputs(x, y)
GuardSet.from_signature(signature)
GraphCache()
compiled = compile_graph(fn)
compiled(*inputs)
compiled.stats()
compiled.explain(*inputs)
```

## 6. Required correctness proof

```text
A1 shape S1 → MISS → compile graph0
A2 shape S1 → HIT graph0
B1 shape S2 → MISS → compile graph1
A3 shape S1 → HIT graph0
```

Expected:

```text
compile_count = 2
cache_entries = 2
hits = 2
misses = 2
```

Outputs and input gradients must match eager execution.

## 7. Non-goals

Not part of v0.12:

- Python bytecode/frame interception;
- symbolic shapes;
- automatic graph partitioning;
- production Dynamo guards;
- AOTAutograd;
- lowering/fusion/code generation.

## 8. Next version

v0.13 will use the now-explicit Graph IR and specialization/cache boundary to study **AOTAutograd-like graph differentiation**.
