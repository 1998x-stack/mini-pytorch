# mini-pytorch v0.15

## Version identity

- **Branch:** `v0.15`
- **Previous version:** `v0.14`
- **Theme:** Runtime / Performance Mechanics
- **Status:** active development
- **Package version target:** `0.15.0`
- **Starting regression baseline:** **221 tests / OK**

v0.15 asks:

> Once a correct `LoweredProgram` exists, how should runtime execution be measured, how long are Values actually live, and how can future memory reuse be based on explicit evidence rather than guesswork?

## Core dependency

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
benchmark / profiling
  ↓
Value lifetime analysis
  ↓
Buffer planning
  ↓
ExecutionPlan
  ↓
PlannedExecutor
```

## Stage 1 — measurement foundation

The first implementation stage contains only:

```text
BenchmarkConfig / BenchmarkReport
ExecutionEvent / ExecutionProfile
profile_lowered(...)
ValueLifetime / analyze_value_lifetimes(...)
```

No buffer reuse is allowed before these measurements and lifetime rules are explicit.

### Benchmark contract

- warmup is explicit and excluded from samples;
- repeated measured runs are required;
- timing uses a monotonic high-resolution clock;
- report per-iteration samples, median, min/max and mean;
- no unit test may rely on absolute speed thresholds;
- a benchmark result is evidence, not a correctness oracle.

### Profiling contract

- profiling is opt-in;
- numerical semantics must equal ordinary `LoweredExecutor` execution;
- one event is recorded per lowered instruction;
- event structure is deterministic even though elapsed times vary;
- ExternalOp and PointwiseFusion remain distinguishable.

### Lifetime contract

For every lowered Graph Value:

```text
ValueLifetime
├── value_id
├── birth_instruction
├── last_use_instruction
├── is_input
├── is_constant
├── is_output
├── is_retained
├── shape
├── dtype
└── device
```

Conventions:

- graph inputs/constants are born before instruction 0 (`birth=-1`);
- node-produced Values are born at the instruction that produces them;
- graph outputs stay live through the execution boundary (`last_use = instruction_count`);
- explicitly retained/saved Values also stay live through that boundary;
- internal fused Values are still represented in the educational analysis because the current reference fusion executor materializes them for inspectability/AOT saved-value access.

## Later stages

After Stage 1 is validated:

```text
BufferSpec / BufferSlot
      ↓
BufferPlanner
      ↓
ExecutionPlan
      ↓
PlannedExecutor
      ↓
allocation + peak-live metrics
      ↓
reproducible executor benchmark matrix
```

## Correctness oracle

Performance/runtime work never replaces existing semantic references:

```text
GraphInterpreter
      ==
LoweredExecutor
      ==
future PlannedExecutor
```

For AOT gradients:

```text
eager AutogradEngine
      ==
v0.13 backward Graph
      ==
v0.14 lowered backward Graph
      ==
future planned backward execution
```

## Non-goals

- CUDA/Triton code generation;
- GPU benchmarking;
- production allocator;
- asynchronous scheduling;
- multithreaded executor;
- symbolic-shape memory planning;
- claiming a speedup from instruction-count reduction;
- absolute timing assertions in CI.

## Definition of Done

v0.15 freezes only after:

1. reproducible benchmark primitives exist;
2. optional per-instruction profiling exists;
3. Value lifetimes are explicit and tested;
4. conservative buffer planning exists;
5. an inspectable `ExecutionPlan` exists;
6. a `PlannedExecutor` matches existing reference execution;
7. AOT saved Values survive forward planning correctly;
8. allocation/reuse/peak-live metrics are exposed;
9. an executor comparison example exists;
10. all historical and new tests pass on exact branch HEAD.
