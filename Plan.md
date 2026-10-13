# mini-pytorch — Architecture-Driven Development Plan

> Canonical roadmap for `main`.
>
> **Invariant:** `main` is always the newest integrated, validated framework state. Each version branch is a readable historical learning snapshot.

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
Runtime / Performance
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
| v0.13 | AOTAutograd-like | explicit backward Graph | **206** |

## v0.13 completed architecture

```text
forward Graph
      ↓
differentiate_graph
      ↓
DifferentiationSpec
      ↓
explicit saved forward Values
      ↓
backward Graph
      ↓
GraphInterpreter / Dispatcher
      ↓
input gradients
```

Validated VJP set:

```text
add
mul
matmul (2-D)
relu
sum
```

Validated properties:

- eager vs AOT forward equality;
- eager AutogradEngine vs backward-Graph gradient equality;
- broadcasting reduction;
- reduction axis/keepdims;
- shared-gradient accumulation;
- explicit non-scalar grad seeds;
- CPU/EDU placement;
- unsupported-op error boundary;
- explicit first-order/no-higher-order boundary.

---

# 3. Current next milestone — v0.14 Lowering / Fusion / Inductor-like Concepts 🎯

## 3.1 Why lowering follows AOT graphs

The project now has explicit programs for both directions:

```text
forward Graph
backward Graph
```

They are inspectable and executable, but still resemble the high-level logical operator stream captured from eager execution.

The next compiler question is:

> Can we transform a correct high-level Graph into a simpler/lower-level execution plan without changing its semantics?

This is the conceptual role of lowering/optimization/fusion systems such as TorchInductor, without claiming production parity.

## 3.2 Target architecture

```text
              Graph
                │
                ▼
          GraphAnalysis
        use-count / liveness
                │
                ▼
          normalized Graph
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
      DCE   const fold  fusion analysis
       └────────┼────────┘
                ▼
          LoweredProgram
                │
                ▼
          LoweredExecutor
                │
                ▼
              Tensor
```

The original GraphInterpreter remains a reference executor.

---

## 3.3 v0.14.1 — Graph clone + normalization boundary

Introduce an explicit transform entry point that does not accidentally mutate the source Graph:

```python
normalized = normalize_graph(graph)
```

Requirements:

- deterministic Value/Node remapping;
- preserve input/output ordering;
- preserve constant payloads and metadata;
- validate before and after;
- stable textual representation.

Tests:

- source Graph unchanged;
- cloned program executes equivalently;
- deterministic formatting.

---

## 3.4 v0.14.2 — Use-count and liveness analysis

Build inspectable analysis:

```text
Value id
├── producer
├── users
├── use_count
└── is_graph_output
```

Possible API:

```python
analysis = analyze_graph(graph)
```

This becomes the dependency for DCE and safe fusion grouping.

Tests:

- branched graph;
- shared Value;
- constants;
- graph outputs always live.

---

## 3.5 v0.14.3 — Dead-value / dead-node elimination

Captured eager code may compute a Tensor that is never returned or consumed by live output dataflow.

Implement:

```text
outputs
  ↓ reverse reachability
live Values / Nodes
  ↓
new Graph
```

Rules:

- only remove operations currently modeled as pure;
- never remove graph inputs/outputs incorrectly;
- preserve deterministic ids in rebuilt graph;
- do not claim general side-effect analysis.

Core test:

```python
def fn(x):
    dead = x * x
    live = x + x
    return live
```

Lowered/optimized graph should omit the dead multiply while remaining numerically identical.

---

## 3.6 v0.14.4 — Safe constant folding

Fold only operations whose complete inputs are captured constants and whose semantics are pure/stable.

Initial candidate set:

```text
add
mul
relu
```

Possible process:

```text
constant inputs
      ↓
reference execution
      ↓
new TensorConstant
      ↓
remove folded node
```

Do not fold device- or runtime-sensitive behavior without an explicit policy.

Tests:

- full constant expression;
- constant feeding live runtime op;
- no fold when runtime Value is involved;
- CPU/EDU policy documented.

---

## 3.7 v0.14.5 — Fusion eligibility analysis

Start with deliberately narrow pointwise fusion:

```text
aten::add.Tensor
aten::mul.Tensor
aten::relu.default
```

Treat as initial barriers:

```text
matmul
sum
conv/pool
AOT helper reductions/transposes
```

A fusion group should be explicit data:

```text
FusionGroup
├── input Value ids
├── ordered Node ids
├── output Value ids
└── reason/eligibility metadata
```

Do not merely label adjacent nodes; respect actual dependency/use-count structure.

---

## 3.8 v0.14.6 — Lowered pointwise representation

Introduce a small representation below high-level Graph nodes, e.g.:

```text
LoweredProgram
├── ExternalOp
└── PointwiseFusion
```

A `PointwiseFusion` may contain an expression DAG such as:

```text
out = relu((x * y) + x)
```

rather than three independent Dispatcher calls.

The representation must remain inspectable and deterministic.

---

## 3.9 v0.14.7 — Reference LoweredExecutor

Implement a readable executor for `LoweredProgram`.

Initial educational behavior may use NumPy for a fused pointwise group, but architecture must keep clear separation:

```text
Graph transformation policy
      ≠
numerical execution mechanism
```

The executor should preserve device identity for CPU/EDU according to an explicit policy.

No generated C++/CUDA/Triton is required.

---

## 3.10 v0.14.8 — Forward and backward graph lowering

The same lowering pipeline should accept either:

```text
AOTProgram.forward_graph
```

or:

```text
AOTProgram.backward_graph
```

This is important: v0.13 exists so backward is now just another explicit program that compiler passes can inspect.

Core integration proof:

```text
eager reference
      ==
GraphInterpreter(forward/backward)
      ==
LoweredExecutor(lowered forward/backward)
```

for the supported subset.

---

## 3.11 v0.14.9 — Pass pipeline and explainability

Expose something like:

```python
result = lower_graph(graph)
print(result.stats())
print(result.explain())
```

Suggested metrics:

```text
original_nodes
live_nodes_after_dce
constants_folded
fusion_groups
fused_nodes
external_nodes
lowered_instructions
```

Important:

```text
fewer nodes
≠ automatically faster
```

Performance measurement belongs mainly to v0.15.

---

## 3.12 v0.14.10 — Failure semantics

Explicit errors for:

- malformed Graph;
- unsupported lowerable expression;
- invalid fusion group;
- missing Value dependency;
- dtype/device inconsistency;
- executor input-count mismatch.

Unsupported nodes should remain `ExternalOp` when safe, rather than being silently rewritten incorrectly.

---

## 3.13 v0.14 tests

Minimum families:

### Analysis

- producer/users/use-count;
- shared Values;
- output liveness.

### Transform

- clone/normalization;
- DCE;
- safe constant folding;
- source Graph remains unchanged.

### Fusion

- pointwise chain forms one group;
- branch/use-count prevents unsafe fusion;
- reduction/matmul barriers;
- deterministic grouping.

### Execution

- eager vs GraphInterpreter vs LoweredExecutor output equality;
- CPU/EDU;
- AOT forward Graph;
- AOT backward Graph / input-gradient equality;
- all historical 206 tests remain green.

---

## 3.14 v0.14 Definition of Done

v0.14 is complete when:

- analysis representation is explicit;
- normalization does not mutate source Graph;
- DCE works for pure dead nodes;
- safe constant folding works for documented subset;
- pointwise fusion groups are explicit/deterministic;
- a LoweredProgram exists;
- a reference LoweredExecutor exists;
- forward and AOT backward graphs both lower correctly;
- reference equivalence passes;
- stats/explain surface exists;
- docs/example/README/version/Agents/manifest updated;
- exact final branch-head CI is green;
- frozen v0.14 is integrated into main.

---

# 4. Future milestones

## v0.15 — Runtime / Performance Mechanics

Central question:

> Once optimized/lowered programs exist, what runtime mechanisms actually determine execution cost?

Planned areas:

```text
execution timing / profiling
memory lifetime planning
buffer reuse / simple pool
preallocation
execution plans
compiled-vs-eager measurements
fusion benefit/cost cases
```

Any speed claim must be measured, not inferred from graph node count.

## v1.0 — Integrated Architecture

The final educational architecture should connect end-to-end:

```text
Tensor / Storage
      ↓
Dispatcher / Backend
      ↓
AutogradEngine
      ↓
Module / Optimizer / Data
      ↓
Runtime / Checkpoint
      ↓
Distributed
      ↓
Graph Capture
      ↓
Guards / Cache
      ↓
AOT backward Graph
      ↓
Lowering / Fusion
      ↓
Runtime planning
```

v1.0 means the major architectural boundaries can be followed in one small readable codebase—not feature parity with PyTorch.

---

# 5. Governance checklist

Every future version must include:

```text
README.md
version.md
Agents.md
relevant docs
relevant example
tests
```

Freeze sequence:

```text
exact version-branch HEAD
      ↓
full GitHub Actions regression green
      ↓
freeze version branch
      ↓
fast-forward validated tree into main
      ↓
restore canonical main docs
      ↓
advance Plan.md
```
