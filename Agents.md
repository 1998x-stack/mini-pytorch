# Agents.md — mini-pytorch v0.15 Development Contract

> Read before modifying branch `v0.15`.

## Branch identity

- Branch: `v0.15`
- Previous: `v0.14`
- Theme: Runtime / Performance Mechanics
- Status: release candidate until exact final-head CI passes
- Starting baseline: 221 tests
- Current validated implementation baseline: 247 tests

## Required dependency order

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

Do not make performance claims before measurement and resource semantics are explicit.

## Measurement contract

### Benchmarking

- use a monotonic high-resolution timer;
- warmup count is explicit and excluded;
- repeats and iterations-per-repeat are explicit positive integers;
- report per-iteration samples and median/mean/min/max;
- do not use absolute speed thresholds in unit tests;
- planning/capture/lowering time must remain distinguishable from steady-state execution;
- a slower planned runtime is valid evidence and must not be hidden.

### Profiling

- profiling is opt-in;
- profiling does not change numerical results;
- one event per `LoweredProgram` instruction;
- instruction index/kind/label/input/output Value ids remain inspectable;
- ordinary `LoweredExecutor` execution does not pay per-instruction timing overhead.

## Lifetime contract

Two lifetime views intentionally coexist.

### Instruction-level

`ValueLifetimeAnalysis` describes lowered instruction boundaries.

Rules:

- input/constant birth = -1;
- Graph outputs remain live through execution boundary;
- retained/AOT saved Values remain live through execution boundary;
- fused internal Values remain visible because the reference fusion path materializes them.

### Node-level buffer interval

`BufferInterval` describes optimized Graph node positions so reuse may be reasoned about inside one `PointwiseFusion`.

No buffer reuse before last node-level use.

## Buffer planning rules

A reusable planner slot is allowed only when all are explicit:

```text
producer belongs to PointwiseFusion
not Graph output
not retained/AOT saved Value
exact shape equality
exact dtype equality
exact device equality
non-overlapping lifetimes
```

The first algorithm is deterministic greedy reuse.

Never assign these to a reusable planner slot:

- external inputs;
- captured constants;
- graph outputs;
- retained forward Values;
- ExternalOp outputs.

Buffer assignment is plan data, not hidden mutation.

## PlannedExecutor rules

- `PlannedExecutor` remains first-order and executes under no-grad semantics;
- only add/mul/relu inside `PointwiseFusion` may currently use preallocated `out=` destinations;
- `ExternalOp` always continues through Dispatcher;
- a fresh physical slot pool is allocated for each executor call;
- returned output from an earlier call must not alias a later call's slot pool;
- explicitly requested intermediate Values must have been retained in the `ExecutionPlan`;
- CPU/EDU placement must remain correct;
- runtime metrics are inspectable separately from wall-clock timing.

## AOT integration contract

Forward planning must preserve every v0.13 saved Value required by backward.

Required chain:

```text
AOT source saved ids
      ↓
LoweringResult.mapped_value_id
      ↓
ExecutionPlan(retained lowered ids)
      ↓
PlannedExecutor.run_with_values
      ↓
planned backward executor
```

Correctness oracle:

```text
GraphInterpreter
      ==
LoweredExecutor
      ==
PlannedExecutor
```

and:

```text
eager Autograd gradients
      ==
AOT reference gradients
      ==
planned backward gradients
```

## Capture identity invariant

During an active capture, every observed Tensor object must remain strongly referenced by the session while object identity is used for the runtime mapping.

Do not regress to a bare:

```text
id(tensor) -> Value
```

map without lifetime protection. Python object ids may be recycled after GC and can otherwise create multiple producer nodes for one Graph Value.

## Performance-reporting rules

- repeated samples, not single-shot timing;
- median is the primary latency summary;
- min/max/mean remain visible;
- compare identical workload/input structure;
- allocation/reuse/peak-live metrics are separate from wall-clock timing;
- fewer instructions or fewer buffers is not itself a latency claim;
- do not claim PyTorch performance parity.

## Non-goals

- CUDA/Triton/C++ code generation;
- GPU timing;
- production caching allocator;
- async streams/events;
- multithreaded scheduler;
- symbolic-shape planner;
- arbitrary alias/mutation optimization;
- production profiler;
- benchmark cherry-picking.

## Testing contract

Required v0.15 tests cover:

- benchmark validation/call counts/statistics;
- profiler numerical equivalence and event structure;
- instruction-level lifetimes;
- AOT retained lifetime extension;
- exact-spec non-overlapping slot reuse;
- output/retained exclusion from reuse;
- deterministic plan formatting;
- PlannedExecutor CPU/EDU equivalence;
- ExternalOp Dispatcher preservation;
- cross-run slot-pool isolation;
- planned runtime metrics;
- planned AOT forward/backward equivalence;
- capture object-id recycling regression;
- all historical tests.

Run:

```bash
python -m unittest discover -s tests -v
```

Current validated implementation result:

```text
247 tests
OK
```

## Documentation contract

Before freeze review:

```text
README.md
version.md
Agents.md
MANIFEST.txt
docs/RUNTIME_PERFORMANCE.md
examples/18_runtime_performance.py
```

## Frozen branch rule

After status becomes `frozen`, only documentation corrections, tests clarifying promised v0.15 behavior, or narrow fixes inside this version boundary are allowed. New architecture belongs in v1.0 or a later explicitly planned branch.

## Definition of Done

Freeze only after:

- runtime measurement/lifetime/planning layers are complete;
- PlannedExecutor and AOT integration proofs pass;
- docs/example/manifest are complete;
- exact release-candidate HEAD CI is green;
- `version.md` and this file are switched to frozen;
- exact frozen-head CI is green;
- validated tree is integrated into `main`;
- main canonical roadmap advances to v1.0.
