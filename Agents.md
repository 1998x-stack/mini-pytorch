# Agents.md — mini-pytorch v0.15 Frozen Snapshot Contract

> Branch `v0.15` is frozen. Read this file before making any historical correction.

## Branch identity

- Branch: `v0.15`
- Previous: `v0.14`
- Theme: Runtime / Performance Mechanics
- Status: **frozen historical learning snapshot**
- Frozen validation baseline: **247 tests / OK**

## Frozen dependency chain

```text
LoweredProgram
      ↓
measurement
      ↓
Value lifetime analysis
      ↓
MemoryPlan
      ↓
ExecutionPlan
      ↓
PlannedExecutor
      ↓
benchmark evidence
```

## Preserved runtime invariants

### Measurement

- benchmark warmup is explicit and excluded;
- repeats/iterations are explicit;
- median/mean/min/max are inspectable;
- no absolute speed threshold belongs in tests;
- fewer instructions or fewer buffers is not automatically a speedup.

### Profiling

- profiling is opt-in;
- one event per lowered instruction;
- event order/kind/Value ids remain inspectable;
- profiling does not change numerical semantics.

### Lifetime

- input/constant birth occurs before instruction 0;
- Graph outputs survive the execution boundary;
- AOT retained/saved Values survive the forward boundary;
- fused internal Values remain visible in educational analysis.

### Buffer planning

Reusable planner slots require all of:

```text
PointwiseFusion producer
not graph output
not retained/AOT saved
exact shape
exact dtype
exact device
non-overlapping lifetime
```

Inputs, constants, outputs, retained Values and ExternalOp outputs are not silently assigned reusable planner slots.

### Planned execution

- `PlannedExecutor` is first-order / no-grad;
- only pointwise add/mul/relu use planner-owned `out=` destinations;
- ExternalOps still execute through Dispatcher;
- slot pools are fresh per executor call;
- returned outputs cannot be overwritten by later calls;
- explicitly requested internal Values must be retained in the plan.

### AOT integration

```text
AOT source saved ids
      ↓
source_to_lowered mapping
      ↓
retained ExecutionPlan Values
      ↓
planned forward
      ↓
planned backward
```

Required correctness relation:

```text
GraphInterpreter
== LoweredExecutor
== PlannedExecutor
```

and:

```text
eager gradients
== AOT gradients
== planned backward gradients
```

### Capture identity

Do not regress capture to a bare `id(tensor) -> Value` map without retaining object lifetime. During one active capture, observed Tensor objects must remain strongly referenced so Python object-id recycling cannot merge distinct intermediate Values.

## Historical scope

Allowed changes on frozen `v0.15`:

- documentation corrections;
- tests clarifying behavior already promised by v0.15;
- narrow bug fixes strictly inside the v0.15 runtime/performance boundary.

Do **not** add:

- new compiler/runtime subsystems;
- CUDA/Triton codegen;
- production allocator behavior;
- symbolic-shape planning;
- async/multithreaded scheduling;
- v1.0 integration-only APIs.

Future work starts from latest `main`, never from this frozen branch.

## Validation command

```bash
python -m unittest discover -s tests -v
```

Frozen baseline:

```text
247 tests
OK
```

## Historical reading order

```text
version.md
README.md
mini_pytorch/compiler/benchmark.py
mini_pytorch/compiler/profiler.py
mini_pytorch/compiler/runtime_analysis.py
mini_pytorch/compiler/execution_plan.py
tests/test_runtime_performance.py
tests/test_execution_plan.py
docs/RUNTIME_PERFORMANCE.md
examples/18_runtime_performance.py
```

## Next dependency

The next version is **v1.0 — Integrated Architecture** and must branch from the latest canonical `main`, not from frozen `v0.15`.
