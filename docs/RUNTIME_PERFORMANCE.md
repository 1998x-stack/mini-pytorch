# Runtime / Performance Mechanics — v0.15

v0.15 sits between compiler lowering and a future integrated runtime:

```text
Graph
  ↓
Lowering / Fusion                    v0.14
  ↓
LoweredProgram
  ↓
measurement / lifetime / planning   v0.15
  ↓
ExecutionPlan
  ↓
PlannedExecutor
```

The version is deliberately **measurement-first**. A lower instruction count, fewer allocations, or a fused representation is not automatically a wall-clock speedup.

## 1. Four different questions

Keep these separate:

```text
Correctness
  Does the program compute the same values?

Compiler representation
  How many logical/lowered instructions exist?

Memory/resource behavior
  How many buffers are needed, and when can they be reused?

Wall-clock performance
  How long does execution actually take under a stated benchmark protocol?
```

v0.15 exposes all four rather than collapsing them into a single "optimized" label.

---

## 2. Benchmark primitive

```python
config = BenchmarkConfig(
    warmup=5,
    repeats=20,
    iterations_per_repeat=10,
)
report = benchmark(fn, config=config)
```

Protocol:

1. execute `warmup` calls;
2. discard them;
3. for every repeat, execute exactly `iterations_per_repeat` calls;
4. record monotonic `perf_counter_ns()` elapsed time;
5. normalize each sample to per-iteration latency;
6. report median/mean/min/max.

Why no CI speed threshold?

- hosted runners vary;
- Python GC and OS scheduling vary;
- tiny educational kernels are noisy;
- the architecture should remain correct even when an optimization is slower.

Therefore tests validate benchmark **methodology**, not a particular nanosecond number.

---

## 3. Runtime profiling

`profile_lowered()` wraps the same lowered numerical path and records one event per lowered instruction:

```text
ExecutionEvent
├── instruction_index
├── instruction_kind
├── label
├── input_value_ids
├── output_value_ids
└── elapsed_ns
```

For example:

```text
[0] PointwiseFusion fusion0[mul->add->relu]
[1] ExternalOp aten::sum.default
```

Profiling is opt-in. Ordinary `LoweredExecutor` execution does not read a clock per instruction.

---

## 4. Two lifetime granularities

### 4.1 Instruction-level lifetime

`analyze_value_lifetimes()` works at `LoweredProgram` instruction granularity.

For a fusion instruction containing three nodes, all internal values are born during the same lowered instruction index. This view is useful for:

- profiler/executor boundaries;
- graph output lifetime;
- AOT retained/saved values;
- coarse runtime inspection.

### 4.2 Node-level buffer intervals

Actual reuse inside one `PointwiseFusion` needs finer information. `MemoryPlan` therefore derives node positions from the optimized Graph:

```text
node position:  0   1   2   3   4

%v2               [---]
%v3                   [---]
%v4                       [---]
```

This allows a slot to be reused later in the same fusion instruction after the old logical value has had its final node-level use.

The two models answer different questions and intentionally coexist.

---

## 5. Ownership and escape rules

A value cannot be treated as a reusable internal temporary merely because its final local consumer has executed.

### Inputs

Inputs are externally owned. Their Storage is never assigned to a planner slot.

### Constants

Captured constants are persistent read-only values for this runtime model. They are never planner-owned reusable slots.

### Graph outputs

Outputs escape the call:

```text
producer
   ↓
Graph output
   ↓
caller still owns returned Tensor after executor returns
```

Therefore graph outputs are never assigned reusable slots.

### AOT retained values

Forward values required by backward also escape the ordinary forward lifetime:

```text
forward
  ↓
saved Value ----------------┐
                            ↓
                       backward Graph
```

`build_execution_plan(..., retained_value_ids=...)` extends those lifetimes through the forward boundary and excludes them from reuse.

---

## 6. Conservative compatibility

`BufferSpec` contains:

```text
shape
dtype
device
numel
nbytes
```

v0.15 reuse requires exact:

```text
shape equality
AND dtype equality
AND device equality
AND non-overlapping lifetime
```

Why exact shape instead of only equal bytes/numel?

Because the first educational planner should make alias/layout reasoning obvious. A future production allocator may reuse equal-sized blocks for different logical shapes, but that would mix allocation capacity with Tensor layout policy too early.

---

## 7. Greedy slot planning

Conceptually:

```text
for each eligible Value in birth order:
    search existing slots
    if exact BufferSpec matches
       and previous assigned Value died before current birth:
           reuse slot
    else:
           allocate new logical slot
```

A `BufferSlot` records all logical Values assigned to it:

```text
slot0: values=[%v2, %v4, %v6]
slot1: values=[%v3, %v5]
```

Metrics distinguish:

```text
logical reusable Values
physical slots
reuse_count
naive allocation bytes
planned slot bytes
bytes avoided by reuse
peak eligible live bytes
```

These are model-level resource metrics, not RSS measurements.

---

## 8. Why only pointwise fusion temporaries?

`ExternalOp` continues through Dispatcher:

```text
ExternalOp(matmul / sum / ...)
      ↓
Dispatcher
      ↓
backend kernel
```

Those kernels currently allocate their own outputs. v0.15 does not redesign all kernels into `out=` forms.

For `PointwiseFusion`, however, the reference runtime already owns the NumPy expression. v0.15 can narrowly use:

```python
np.add(a, b, out=destination)
np.multiply(a, b, out=destination)
np.maximum(a, 0, out=destination)
```

and construct `destination` from preallocated `Storage.empty(...)`.

This gives real, inspectable reuse without pretending every backend op supports output-buffer execution.

---

## 9. ExecutionPlan

`ExecutionPlan` combines:

```text
what executes                   LoweredProgram
when Values are needed          ValueLifetimeAnalysis
which temporary gets which slot MemoryPlan
when env references can drop    release schedule
```

Example conceptual printout:

```text
ExecutionPlan:
  [0] PointwiseFusion fusion0 nodes=[node0,node1,node2]
  [1] ExternalOp aten::sum.default nodes=[node3]

MemoryPlan:
  slot0 -> %v2, %v4
  slot1 -> %v3
  reuse_count=1
```

The plan is deterministic and inspectable.

---

## 10. PlannedExecutor semantics

`PlannedExecutor` remains first-order and executes under `no_grad()` like `LoweredExecutor`.

### Fusion nodes

If an output has a planned slot:

```text
BufferSlot
  ↓
Storage.empty(...)
  ↓
Tensor._from_storage(...)
  ↓
NumPy out= operation
```

If no slot is allowed (for example graph output or retained Value), normal fresh Tensor allocation is used.

### External ops

External ops still dispatch normally.

### Lifetime releases

After a node reaches the last use of an internal value, that environment reference is dropped. The planner assignment itself is static; runtime release simply makes the lifetime decision visible in execution.

### Cross-call safety

A new slot pool is created for every executor call. Returned values from call A therefore cannot be overwritten by slot reuse in call B.

---

## 11. AOT integration

The central end-to-end proof is:

```text
Python function
   ↓
aot_compile
   ├── forward Graph
   └── backward Graph

forward Graph
   ↓
lower_graph
   ↓
source saved Value id → lowered Value id
   ↓
build_execution_plan(retained=saved ids)
   ↓
PlannedExecutor.run_with_values
   ↓
saved Tensors

backward Graph
   ↓
lower_graph
   ↓
build_execution_plan
   ↓
PlannedExecutor
   ↓
input gradients
```

The expected equality chain is:

```text
eager Autograd
== AOT reference
== lowered backward
== planned backward
```

---

## 12. Capture identity bug found by runtime stress tests

Long pointwise workloads create many short-lived eager temporary Tensors. Older capture code stored only:

```text
id(tensor) → Graph Value
```

If a temporary died, CPython could recycle its object id during the same capture. A later Tensor could then incorrectly look like the old Tensor and reuse its Graph Value, producing an invalid graph with multiple producer nodes.

v0.15 fixes the session invariant by retaining observed Tensor objects strongly for the full capture and validating both:

```text
same id
AND same object (`is`)
```

A dedicated regression test repeatedly creates/deletes short-lived temporaries and verifies every captured node has a unique output Value.

---

## 13. Benchmark matrix

`examples/18_runtime_performance.py` compares:

```text
Eager Tensor execution
GraphInterpreter
LoweredExecutor
PlannedExecutor
```

for two workloads:

```text
pointwise-heavy:
  mul → add → relu → mul → add → relu → sum

barrier-heavy:
  matmul → relu → sum
```

Interpretation rules:

- compare identical input shapes and executor work;
- capture/lowering/planning happen before steady-state timing;
- median is the headline latency, not a single sample;
- also inspect fusion/instruction/allocation metrics;
- if PlannedExecutor is slower, report that result rather than hiding it.

---

## 14. What this is not

This is not a production memory allocator or TorchInductor runtime. There is no:

- CUDA/Triton allocation planning;
- stream-aware lifetime;
- asynchronous free/reuse;
- cross-shape capacity allocator;
- mutation/alias analysis sufficient for arbitrary in-place graphs;
- symbolic dynamic-shape planner;
- multithreaded task scheduler;
- promise of PyTorch-level speed.

The learning goal is the boundary:

```text
compiler says what to execute
runtime says how to schedule resources
measurement tells whether the choice helped
```
