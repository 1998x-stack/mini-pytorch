# mini-pytorch — Architecture-Driven Development Plan

> Canonical roadmap for `main`.
>
> **Invariant:** `main` is always the newest integrated, validated framework state. Every version branch is a readable historical learning snapshot.

## 1. Project dependency chain

```text
Tensor / Dynamic Autograd
      ↓
Module / Optimizer
      ↓
Data pipeline
      ↓
Tensor memory model
      ↓
Dispatcher / Backend
      ↓
Advanced Autograd Engine
      ↓
Training Runtime
      ↓
Distributed / DDP
      ↓
Graph IR
      ↓
Guards / Graph Cache
      ↓
AOT-style Graph Differentiation
      ↓
Lowering / Fusion
      ↓
Runtime / Performance Mechanics
      ↓
Integrated architecture
```

One version introduces one dominant mental model.

---

# 2. Completed milestones

| Version | Theme | Key validated mechanism | Tests |
|---|---|---|---:|
| v0.1 | Dynamic Autograd | Tensor → Function → backward | historical |
| v0.2 | Optimizer/Scheduler | groups/state/LR scheduling | historical |
| v0.3 | Data pipeline | Dataset/Sampler/DataLoader | historical |
| v0.4 | Tensor memory | Storage/stride/views/version | historical |
| v0.5 | Dispatcher | schema/keys/redispatch | 80 |
| v0.6 | CNN ops | Conv/Pool/BN/Dropout | 90 |
| v0.7 | Device/Backend | CPU + EDU placement | 103 |
| v0.8 | Autograd Engine | lifetime/hooks/retain_graph | 123 |
| v0.9 | Training Runtime | deterministic checkpoint resume | 140 |
| v0.10 | DDP | reducer/buckets/all-reduce equivalence | 159 |
| v0.11 | Graph IR | eager capture/replay | 176 |
| v0.12 | Guards/Cache | specialization/recompile | 192 |
| v0.13 | AOTAutograd-like | explicit backward Graph | 206 |
| v0.14 | Lowering/Fusion | analysis → DCE/fold/fusion → LoweredProgram | **221** |

## v0.14 completed architecture

```text
Graph
  ↓
normalize_graph
  ↓
analyze_graph
  ↓
DCE
  ↓
safe constant folding
  ↓
use-aware fusion planning
  ↓
LoweredProgram
  ├── ExternalOp
  └── PointwiseFusion
  ↓
LoweredExecutor
```

Validated properties:

- source Graph remains unchanged by passes;
- producer/users/use-count/liveness are explicit;
- unreachable pure branches are removed;
- safe add/mul/relu constant chains can fold;
- grad-requiring constants are not folded;
- pointwise fusion uses dependency and occurrence-level use-count, not adjacency alone;
- shared/branching values stop conservative fusion;
- matmul/reduction/conv/pool/AOT helper ops remain barriers;
- CPU/EDU logical placement is preserved;
- ExternalOps execute through Dispatcher;
- source-to-lowered Value mappings preserve AOT saved-value ABI;
- the same pipeline lowers forward and backward Graphs;
- lowered backward gradients match the v0.13 reference;
- node/instruction reduction is not treated as a performance result.

---

# 3. Current next milestone — v0.15 Runtime / Performance Mechanics 🎯

## 3.1 Why runtime follows lowering

v0.14 provides an explicit lower-level program:

```text
LoweredProgram
├── ExternalOp
└── PointwiseFusion
```

It is correct and inspectable, but the reference runtime still:

- materializes Tensor intermediates;
- has no explicit allocation plan;
- has no Value lifetime intervals;
- has no reusable buffer slots;
- has no per-instruction profiling;
- has no reproducible benchmark framework.

The next question is:

> Once execution structure is explicit, how can the runtime measure where time/memory goes and safely reduce runtime overhead based on evidence?

The order matters:

```text
measure
  ↓
model lifetime/allocation
  ↓
plan reuse
  ↓
execute
  ↓
measure again
```

Not:

```text
guess optimization
  ↓
claim speedup
```

---

## 3.2 v0.15.1 — Reproducible benchmark/timing primitive

Introduce a small timing layer, for example:

```text
BenchmarkConfig
├── warmup
├── repeats
├── iterations_per_repeat
└── synchronization policy

BenchmarkSample
BenchmarkReport
```

Required semantics:

- warmup iterations excluded from statistics;
- same callable and inputs per comparison;
- repeated samples;
- report median plus range/percentiles where practical;
- distinguish setup/planning/compile time from steady-state execution;
- CPU timing is the performance reference; EDU remains a logical placement test.

Suggested API:

```python
report = benchmark(fn, *inputs, warmup=5, repeats=20)
print(report.median_ms)
```

Do not benchmark inside unit tests using fragile absolute time thresholds.

---

## 3.3 v0.15.2 — Runtime profiling events

Add optional instrumentation around `LoweredProgram` execution.

Suggested model:

```text
ExecutionEvent
├── instruction_index
├── instruction_kind
├── operator/fusion label
├── elapsed_ns
├── input Value ids
├── output Value ids
└── allocation events

ExecutionProfile
├── events
├── total_elapsed_ns
└── summary()
```

Requirements:

- profiling optional;
- instrumentation must not alter numerical results;
- profile output deterministic in structure, timing values naturally variable;
- separate `ExternalOp` and `PointwiseFusion` timing.

---

## 3.4 v0.15.3 — Value lifetime analysis

Use the explicit instruction stream to derive lifetime intervals.

For each lowered Value:

```text
ValueLifetime
├── value_id
├── birth_instruction
├── last_use_instruction
├── is_input
├── is_constant
├── is_output
├── shape
├── dtype
└── device
```

Mental model:

```text
instruction:  0   1   2   3   4

value A:      [-----------]
value B:          [---]
value C:              [-------]
```

A buffer can only be reused after the old value's last use.

Special rules:

- graph inputs are externally owned;
- constants are persistent/read-only for this runtime model;
- graph outputs escape the execution and cannot be reused before return;
- saved forward Values required by AOT backward also escape the forward execution boundary and must be marked as retained.

That last point is critical: runtime memory planning must preserve the v0.13 saved-value ABI.

---

## 3.5 v0.15.4 — Buffer abstraction

Introduce a small runtime buffer model rather than reusing arbitrary Tensor objects implicitly.

Suggested concepts:

```text
BufferSpec
├── shape / numel
├── dtype
├── device
└── bytes

BufferSlot
├── slot_id
├── spec
└── assigned Value ids
```

Compatibility starts conservatively:

```text
same dtype
same device
compatible shape/numel
non-overlapping lifetimes
not externally owned
```

Do not add clever aliasing before the simple model is proven correct.

---

## 3.6 v0.15.5 — BufferPlanner

Input:

```text
LoweredProgram
+
ValueLifetime[]
```

Output:

```text
BufferPlan
├── value_to_slot
├── slots
├── allocation_count
├── reuse_count
└── estimated_peak_bytes
```

First algorithm should be readable greedy reuse:

```text
for Values in birth order:
    release slots whose last-use is before current birth
    find compatible free slot
    else allocate new slot
```

Tests should prove:

- overlapping lifetimes never share a slot;
- non-overlapping compatible values may share;
- dtype/device mismatch blocks reuse;
- input/constant/output ownership rules are respected;
- retained AOT saved Values remain alive through the required boundary.

---

## 3.7 v0.15.6 — ExecutionPlan

Combine compiler output and runtime decisions:

```text
ExecutionPlan
├── LoweredProgram
├── ValueLifetime[]
├── BufferPlan
├── input bindings
├── output bindings
└── retained/saved bindings
```

This becomes the stable handoff from compiler lowering to runtime execution.

The key conceptual split is:

```text
LoweredProgram
= what operations execute

ExecutionPlan
= how runtime resources are assigned while executing them
```

---

## 3.8 v0.15.7 — PlannedExecutor

Implement a reference runtime that consumes `ExecutionPlan`.

Requirements:

- numerical outputs equal `LoweredExecutor`;
- reuse decisions visible through buffer slot ids/counters;
- CPU first;
- EDU logical placement correctness remains tested;
- no unsafe mutation of user inputs/constants;
- graph outputs receive stable returned Tensor values;
- AOT saved forward Values remain available for lowered backward execution.

If output-buffer forms make specific pointwise operations readable, add them narrowly. Do not redesign every backend kernel in this version.

---

## 3.9 v0.15.8 — Allocation / peak-live metrics

Expose measurements distinct from time:

```text
logical Value count
physical BufferSlot count
allocation count
reuse count
estimated peak live bytes
bytes avoided/reused
```

Central memory proof example:

```text
logical intermediates = 5
physical compatible slots = 2
```

with exact numerical equivalence.

Again, fewer allocations is a measured property; wall-clock speed must be benchmarked separately.

---

## 3.10 v0.15.9 — Executor benchmark matrix

Build a reproducible educational comparison:

```text
Eager Python/Tensor execution
GraphInterpreter
LoweredExecutor
PlannedExecutor
```

Use at least two workload families:

### Pointwise-heavy

```text
mul → add → relu → mul → add
```

Useful for fusion/runtime-overhead study.

### Barrier-heavy

```text
matmul → relu → sum
```

Useful to show that fusion opportunities and runtime behavior differ by graph structure.

For each report:

```text
input shape
warmup
repeat policy
median time
range/percentile
allocation metrics
fusion count
instruction count
```

No assertion that the educational runtime must beat eager NumPy/Python.

A slower result is still a valid architectural finding.

---

## 3.11 v0.15.10 — AOT forward/backward runtime integration

Run the full compiler/runtime chain:

```text
Python function
   ↓
Graph capture
   ↓
AOT forward/backward graphs
   ↓
lowering
   ↓
ExecutionPlan(forward)
ExecutionPlan(backward)
   ↓
PlannedExecutor
```

Required proof:

```text
eager output
== GraphInterpreter output
== LoweredExecutor output
== PlannedExecutor output
```

and:

```text
eager gradients
== AOT gradients
== lowered-backward gradients
== planned-backward gradients
```

The forward plan must retain exactly the saved Values required by backward.

---

## 3.12 v0.15.11 — Explain / profiling report

Provide a human-readable view such as:

```text
ExecutionPlan
  instruction 0: PointwiseFusion
    produces %v3 -> buffer0
  instruction 1: ExternalOp(sum)
    reads %v3
    produces %v4 -> buffer1
    releases buffer0

Buffers
  buffer0: float64 cpu 1024 elements
    values: %v3, %v7
  buffer1: ...

Metrics
  logical_values=8
  slots=3
  reuse_count=4
  estimated_peak_bytes=...
```

The runtime should be explainable before it is optimized further.

---

## 3.13 v0.15 failure semantics

Fail explicitly for:

- malformed lifetime intervals;
- buffer compatibility violations;
- attempts to reuse externally owned input/constant storage;
- missing Value/slot bindings;
- device/dtype mismatch;
- retained AOT saved Value released too early;
- unsupported planned instruction form;
- benchmark configuration with non-positive repeats/iterations.

No silent fallback that invalidates reported profiling/allocation metrics.

---

# 4. v0.15 Definition of Done

v0.15 freezes only when all are true:

1. benchmark methodology is explicit and reproducible;
2. optional per-instruction profiling exists;
3. Value lifetimes are explicit and inspectable;
4. conservative buffer planning exists;
5. `ExecutionPlan` separates compiler work from runtime resource assignment;
6. `PlannedExecutor` matches existing reference executors numerically;
7. AOT saved Values remain correct across forward/backward planning;
8. allocation/reuse/peak-live metrics are tested;
9. eager/Graph/lowered/planned benchmark example exists;
10. no speed claim is made without measured evidence;
11. all historical tests plus new tests are green;
12. docs/example/manifest/version/Agents are complete;
13. exact final branch-head CI is green.

---

# 5. v1.0 — Integrated Architecture

After v0.15, the project should stop adding major mechanisms and consolidate the complete educational stack.

Target map:

```text
Python API
   ↓
Tensor / Storage / Device
   ↓
Dispatcher / Backend
   ↓
AutogradEngine
   ↓
Module / Optimizer / Data
   ↓
Runtime / Checkpoint / DDP
   ↓
Capture / Graph IR
   ↓
Guards / Cache
   ↓
AOT forward + backward Graphs
   ↓
Lowering / Fusion
   ↓
ExecutionPlan / Runtime profiling / Buffer planning
```

v1.0 tasks should focus on:

- architecture documentation consistency;
- end-to-end learning map;
- one integrated example from eager training to graph/runtime execution;
- API cleanup without erasing historical branch clarity;
- comparison table mapping mini-pytorch concepts to real PyTorch subsystems;
- final full regression and release snapshot.

---

# 6. Engineering sequence

Recommended implementation order from current `main`:

```text
v0.14 Lowering / Fusion                     ✅
      ↓
v0.15 Runtime / Performance Mechanics       🎯
      ↓
v1.0  Integrated Architecture
```

Do not skip directly to low-level optimization before lifetime/profiling/resource semantics are explicit.
