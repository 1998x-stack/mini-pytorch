# mini-pytorch v0.15

## Version identity

- **Branch:** `v0.15`
- **Previous version:** `v0.14`
- **Theme:** Runtime / Performance Mechanics
- **Status:** frozen historical learning snapshot
- **Package version:** `0.15.0`
- **Starting baseline:** 221 tests
- **Frozen validation baseline:** **247 / 247 tests passed**

v0.15 answers:

> Once a correct `LoweredProgram` exists, how should runtime execution be measured, how long are Values live, how can temporary storage be safely reused, and how do we benchmark the result without making unsupported performance claims?

## Core delta from v0.14

```text
v0.14
Graph
  ↓
LoweredProgram
  ↓
LoweredExecutor

v0.15
LoweredProgram
  ↓
Benchmark / Profiler
  ↓
Value lifetime analysis
  ↓
Buffer intervals / MemoryPlan
  ↓
ExecutionPlan
  ↓
PlannedExecutor
```

## Added in this version

### Measurement

- `BenchmarkConfig`
- `BenchmarkReport`
- `benchmark()`
- explicit warmup/repeats/iterations-per-repeat
- median/mean/min/max latency summaries
- no absolute speed assertions in tests

### Profiling

- `ExecutionEvent`
- `ExecutionProfile`
- `ProfiledResult`
- `profile_lowered()`
- one event per lowered instruction
- deterministic instruction structure with naturally variable timing

### Runtime lifetime analysis

- `ValueLifetime`
- `ValueLifetimeAnalysis`
- `analyze_value_lifetimes()`
- input/constant birth before instruction 0
- graph output boundary extension
- AOT retained/saved-value boundary extension
- fused internal Value visibility

### Buffer planning

- `BufferSpec`
- `BufferInterval`
- `BufferSlot`
- `MemoryPlan`
- exact shape/dtype/device compatibility
- non-overlap requirement
- conservative eligibility limited to pointwise-fusion temporaries
- graph output and retained values excluded from reuse
- input/constant and ExternalOp output ownership preserved

### Execution planning

- `ExecutionStep`
- `ExecutionPlan`
- `build_execution_plan()`
- explicit release-after-node schedule
- separation of compiler work (`LoweredProgram`) from runtime resource assignment (`ExecutionPlan`)

### Planned execution

- `PlannedExecutor`
- preallocated `Storage.empty()` slots for eligible pointwise temporaries
- NumPy `out=` execution for add/mul/relu when a slot is assigned
- ExternalOps continue through Dispatcher
- fresh slot pool per call
- `run_with_stats()`
- `run_with_values()` for explicitly retained values

### Runtime metrics

- physical slot count
- logical assigned Value count
- reuse count
- naive allocation bytes
- planned buffer bytes
- bytes avoided by reuse
- peak eligible live bytes
- released environment reference count

### Compiler robustness fix

Runtime stress tests exposed an inherited capture identity bug. A capture session previously stored only `id(tensor) -> Value`. Short-lived temporary Tensors could be garbage-collected and their Python object ids recycled, causing distinct intermediates to share one Graph Value and creating multiple producers.

Capture now keeps strong references to all observed Tensor objects for the full session and validates object identity with `is` before reusing a Value mapping.

A dedicated regression test creates/deletes many short-lived temporaries and verifies unique producer Values.

## Frozen architecture

```text
Python eager
   ↓
Graph capture
   ↓
AOT forward/backward Graphs
   ↓
Lowering / Fusion
   ↓
LoweredProgram
   ├── ExternalOp
   └── PointwiseFusion
   ↓
Runtime measurement
   ├── benchmark
   └── profiling
   ↓
Value lifetime analysis
   ↓
MemoryPlan
   ├── BufferInterval
   └── BufferSlot
   ↓
ExecutionPlan
   ↓
PlannedExecutor
```

## Correctness proofs

```text
GraphInterpreter output
      ==
LoweredExecutor output
      ==
PlannedExecutor output
```

AOT chain:

```text
eager Autograd gradients
      ==
AOT reference gradients
      ==
lowered backward gradients
      ==
planned backward gradients
```

CPU and EDU logical placement are both covered where applicable.

## Buffer reuse rules

A Value is eligible only when:

```text
produced inside PointwiseFusion
AND not Graph output
AND not retained/AOT saved
AND exact shape match
AND exact dtype match
AND exact device match
AND assigned lifetimes do not overlap
```

First planner algorithm:

```text
Values in birth order
   ↓
scan existing compatible slots
   ↓
previous lifetime ended before current birth?
   ├── yes → reuse
   └── no  → create new slot
```

## Important source files

```text
mini_pytorch/compiler/benchmark.py
mini_pytorch/compiler/profiler.py
mini_pytorch/compiler/runtime_analysis.py
mini_pytorch/compiler/execution_plan.py
mini_pytorch/compiler/capture.py
```

## Important tests

```text
tests/test_runtime_performance.py
tests/test_execution_plan.py
tests/test_capture_identity.py
```

## Documentation / example

```text
docs/RUNTIME_PERFORMANCE.md
examples/18_runtime_performance.py
```

## Validation

Validated release-candidate commit:

```text
3d5d8b04e5931fa8dac3b8f37ff3a63a16e54980
```

GitHub Actions environment:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

Result:

```text
Ran 247 tests
OK
```

The frozen branch head differs only by this status/governance documentation and is required to pass its own exact-head CI before integration into `main`.

## Learning goals

After v0.15, the reader should be able to explain:

1. why profiling and benchmarking are different;
2. why instruction reduction is not automatically a speedup;
3. how a Value lifetime constrains memory reuse;
4. why graph outputs and saved forward Values escape normal temporary lifetime;
5. how logical Values can map to fewer physical buffer slots;
6. why `LoweredProgram` and `ExecutionPlan` are different abstractions;
7. why output-buffer execution is introduced narrowly rather than by bypassing Dispatcher;
8. how AOT saved Values constrain forward memory planning;
9. why performance claims require repeated measurement;
10. how runtime stress tests can expose compiler identity/lifetime bugs.

## Explicit non-goals

- production allocator;
- arbitrary equal-byte cross-shape reuse;
- alias/mutation optimization for arbitrary programs;
- CUDA/Triton codegen;
- GPU benchmarking;
- asynchronous streams/events;
- multithreaded scheduling;
- symbolic-shape memory planning;
- production profiler;
- PyTorch performance parity;
- claim that PlannedExecutor is always faster.

## Frozen-branch rule

Future major mechanisms do not belong on `v0.15`. Only documentation corrections, tests clarifying promised v0.15 behavior, or narrow fixes inside this version boundary are allowed.

## Next version

**v1.0 — Integrated Architecture**.

The focus shifts from adding mechanisms to consolidation: architecture consistency, end-to-end examples, API cleanup, mapping to real PyTorch subsystems, and a final learning-oriented release snapshot.
