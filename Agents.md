# Agents.md — mini-pytorch v0.15 Development Contract

> Read before modifying branch `v0.15`.

## Branch identity

- Branch: `v0.15`
- Previous: `v0.14`
- Theme: Runtime / Performance Mechanics
- Status: active until exact final-head CI passes
- Starting baseline: 221 tests

## Required dependency order

```text
LoweredProgram
      ↓
measurement
      ↓
Value lifetime analysis
      ↓
Buffer planning
      ↓
ExecutionPlan
      ↓
PlannedExecutor
      ↓
benchmark evidence
```

Do not implement reuse or make performance claims before measurement/lifetime semantics are explicit.

## Stage 1 contract

### Benchmarking

- use a monotonic high-resolution timer;
- warmup count is explicit and excluded;
- measured repeats and iterations-per-repeat are explicit positive integers;
- report per-iteration samples and summary statistics;
- do not use absolute speed thresholds in unit tests;
- benchmark setup/compile/planning time must remain distinguishable from steady-state execution when later compared.

### Profiling

- profiling is opt-in and must not alter numerical results;
- record one event per `LoweredProgram` instruction;
- keep instruction index/kind/label, input/output Value ids and elapsed time inspectable;
- ordinary execution must not pay timing work when no observer/profiler is requested;
- profiling may use `LoweredExecutor` instrumentation but must not duplicate numerical execution semantics in a second executor.

### Value lifetime analysis

For each Graph Value compute:

```text
birth_instruction
last_use_instruction
ownership flags
shape / dtype / device metadata
```

Rules:

- graph input/constant birth = -1;
- graph outputs live through boundary = `len(instructions)`;
- explicitly retained/saved Values also live through boundary;
- intermediate fused node outputs remain analyzable because v0.14 reference fusion materializes them;
- every referenced runtime Value must have a valid producer/input/constant origin;
- malformed producer/lifetime situations fail explicitly.

## Future buffer-planning rules

When Stage 2 begins, reuse is allowed only if all are explicit:

```text
non-overlapping lifetimes
compatible shape/numel
same dtype
same device
not externally owned input/constant/output
not retained for AOT backward
```

Buffer assignment must be plan data, never hidden mutation.

## Correctness references

For any runtime optimization path:

```text
GraphInterpreter
      ==
LoweredExecutor
      ==
PlannedExecutor
```

AOT forward/backward integration remains mandatory.

## Performance-reporting rules

- repeated samples, not single-shot timings;
- median is the primary latency summary;
- also expose min/max/mean or distribution information;
- compare identical inputs and workload structure;
- allocation/peak-live metrics are separate from wall-clock timing;
- a slower result is valid and must not be hidden;
- no PyTorch performance-parity claim.

## Non-goals

- CUDA/Triton/C++ kernel generation;
- GPU timing;
- production allocator;
- asynchronous streams;
- multithreaded scheduler;
- symbolic-shape planner;
- benchmark cherry-picking.

## Testing contract

Required Stage 1 tests:

- benchmark config validation;
- exact warmup/repeat call counts;
- report statistic consistency without absolute latency assertions;
- profiled vs ordinary lowered numerical equivalence;
- event count/order/kind/Value ids;
- lifetime birth/last-use for inputs/intermediates/outputs/constants;
- graph-output boundary extension;
- retained/AOT saved-value boundary extension;
- fused internal Value lifetime handling;
- malformed lifetime/reference failures;
- CPU/EDU structural parity where relevant;
- all 221 historical tests remain green.

Run:

```bash
python -m unittest discover -s tests -v
```

## Definition of Done

Freeze v0.15 only after:

- benchmark/profiling/lifetime layers are stable;
- conservative buffer planning and ExecutionPlan exist;
- PlannedExecutor exists and matches references;
- AOT saved Values remain correct;
- allocation/reuse/peak-live metrics are inspectable;
- docs/example/manifest/version are complete;
- exact final `v0.15` HEAD CI is green;
- validated tree is integrated into `main`.
