# mini-pytorch — Architecture-Driven Development Plan

> Canonical roadmap for `main`.
>
> **Invariant:** `main` is always the newest integrated, validated state. Version branches are readable learning snapshots. A completed version is validated, frozen, then integrated into `main` before the next branch is created.

## 1. Project purpose

mini-pytorch removes one layer of PyTorch “magic” at a time with readable Python/NumPy implementations.

The roadmap follows architectural dependency order:

```text
Tensor / Autograd
      ↓
Module / Optimizer
      ↓
Data pipeline
      ↓
Tensor memory model
      ↓
Operator Dispatcher
      ↓
NN operator stack
      ↓
Device / Backend
      ↓
Advanced Autograd Engine
      ↓
Training Runtime / Reproducibility
      ↓
Distributed / DDP concepts
      ↓
Graph capture / compiler stack
```

One version should introduce one dominant mental model and preserve earlier invariants.

---

# 2. Completed milestones

## v0.1 — Eager Tensor + Dynamic Autograd ✅

```text
Tensor → Function → dynamic graph → backward → Parameter → Module → Optimizer
```

Key mechanisms:

- Tensor / requires_grad;
- reverse-mode autodiff;
- dynamic graph;
- gradient accumulation;
- Module / Parameter / Linear / Sequential;
- SGD / Adam;
- numerical gradcheck.

## v0.2 — Optimizer State + LR Scheduler ✅

```text
Parameter
   ↓
Optimizer parameter groups + state
   ↓
LR Scheduler
```

Key mechanisms:

- enhanced SGD;
- Adam / AdamW / Adamax;
- RMSprop / Adagrad;
- parameter groups;
- optimizer state serialization;
- StepLR / MultiStepLR / ExponentialLR;
- CosineAnnealingLR / LinearLR / LambdaLR;
- ReduceLROnPlateau.

## v0.3 — Dataset / Sampler / DataLoader + MNIST ✅

```text
Dataset → Sampler → BatchSampler → DataLoader → Tensor batch
```

Key mechanisms:

- Dataset / TensorDataset;
- SequentialSampler / RandomSampler;
- BatchSampler;
- DataLoader;
- MNIST IDX reader;
- Flatten / CrossEntropyLoss;
- MLP MNIST training;
- Loguru / metrics / plots / checkpoints in examples.

## v0.4 — Tensor Memory Model ✅

```text
Storage + shape + stride + storage_offset + VersionCounter
```

Key mechanisms:

- explicit Storage;
- views / aliasing;
- transpose / permute / slicing;
- `view()` vs `reshape()`;
- contiguous semantics;
- inplace mutation;
- shared version counter;
- saved-version Autograd safety.

## v0.5 — Operator Registry + Dispatcher ✅

```text
Tensor API
   ↓
Operator Schema
   ↓
Dispatcher / DispatchKeySet
   ↓
Autograd / NoGrad / backend key
   ↓
redispatch
```

Key mechanisms:

- OperatorSchema;
- operator/kernel registry;
- DispatchKey / DispatchKeySet;
- dispatch priority;
- fallbacks;
- redispatch;
- CPU numerical kernels;
- Autograd wrappers above backend kernels.

Validated baseline: **80 tests**.

## v0.6 — Neural-Network Operator Stack / CNN ✅

```text
nn.Module
    ↓
functional operator
    ↓
Dispatcher
    ↓
Autograd wrapper
    ↓ redispatch
backend NN kernel
```

Key mechanisms:

- Conv2d reference loops + im2col;
- Conv2d input/weight/bias gradients;
- MaxPool2d;
- Dropout;
- BatchNorm1d / BatchNorm2d;
- Parameter vs Buffer state;
- CNN MNIST example.

Validated baseline: **90 tests**.

## v0.7 — Device / Backend Abstraction ✅

```text
Tensor metadata
     ↓
Storage + Device
     ↓
backend-owned allocation
     ↓
Dispatcher
 ┌────┴────┐
 CPU      EDU
```

Key mechanisms:

- immutable Device;
- Backend abstraction/registry;
- CPU + deterministic educational `edu` backend;
- backend-owned Storage;
- Tensor.device / Tensor.to(device);
- cross-device alias break;
- differentiable transfer;
- device-aware DispatchKey inference;
- mixed-device errors;
- Module.to(device).

Validated baseline: **103 tests**.

## v0.8 — Advanced Autograd Engine ✅

```text
forward graph
     ↓
GradFn / Context
     ↓
AutogradEngine
 ┌────┼───────────────┐
 ▼    ▼               ▼
VJP  graph lifetime  hooks
 │    retain_graph    │
 ▼                    ▼
leaf grad       retained non-leaf grad
```

Implemented:

- explicit `AutogradEngine` separated from Tensor;
- topological traversal and Engine-local gradient accumulation;
- Context release state;
- default graph release after successful backward;
- repeated-backward failure after release;
- `retain_graph=True`;
- non-leaf `retain_grad()`;
- gradient hooks;
- removable hook handles;
- hook gradient transformation;
- shared-node accumulation;
- public custom `Function / Context / GradFn` teaching workflow;
- saved-version safety preserved;
- CPU/EDU parity;
- explicit `create_graph=True` limitation rather than fake higher-order AD.

Validated baseline: **123 tests**.

Architecture docs:

```text
docs/AUTOGRAD_ENGINE.md
examples/11_autograd_engine.py
```

---

# 3. Current next milestone — v0.9 Training Runtime / Serialization / Reproducibility 🎯

## 3.1 Why v0.9 comes now

By v0.8 the framework can already express a realistic training stack:

```text
Dataset / DataLoader
      ↓
Module / Parameters
      ↓
Dispatcher / Backend
      ↓
Autograd Engine
      ↓
Optimizer
      ↓
LR Scheduler
```

But a training process is more than its model weights.

If execution stops after step 731 and only the model parameters are saved, the continuation may differ because the following state can be lost:

```text
optimizer moments / momentum
scheduler epoch/step
random-number generator state
sampler/shuffle progression
current epoch
global step
best metric / runtime metadata
```

The central v0.9 question is:

> What exact state must be captured, validated and restored so a stopped training run can continue with the same future behavior?

The milestone is **training-state architecture**, not merely adding `pickle.dump()`.

---

## 3.2 Target architecture

```text
                    Training process
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
       Module          Optimizer        Scheduler
          │               │                │
          └───────────────┼────────────────┘
                          │
                          ▼
                     TrainingState
              ┌───────────┼────────────┐
              ▼           ▼            ▼
          progress      RNG state    metadata
              │           │            │
              └───────────┼────────────┘
                          ▼
                    CheckpointState
                          │
                  schema validation
                          │
                          ▼
               serializer / deserializer
                          │
                    NPZ + JSON-like
                          │
                          ▼
                     checkpoint
                          │
                          ▼
                         load
                          │
                  validate + restore
                          │
                          ▼
                deterministic continue
```

Runtime state collection should be separate from file-format mechanics.

---

# 3.3 v0.9.1 — Explicit RNG / Seed State

## Problem

Current examples may seed NumPy or individual components, but reproducibility is not a first-class framework concept.

A training checkpoint cannot claim deterministic resume if random choices after load diverge.

## Implement

Introduce a small framework RNG abstraction or explicit RNG-state utilities.

Possible API:

```python
manual_seed(42)
get_rng_state()
set_rng_state(state)
```

The exact implementation should stay small and NumPy-first.

Important distinction:

```text
seed
!=
current RNG state
```

A seed initializes a random stream. A checkpoint must usually restore the **current stream position/state**, not merely replay the original seed.

## Required semantics

- deterministic seed initialization;
- export current RNG state;
- restore state;
- repeated sequence after restore matches exactly;
- state is serializable without opaque Python object assumptions where practical;
- explicit boundary around random sources not controlled by mini-pytorch.

## Scope integration

Review randomness in:

- parameter initialization;
- Dropout;
- RandomSampler / DataLoader shuffle;
- any training example using NumPy random generators.

Do not silently claim process-wide determinism if an external source is not captured.

---

# 3.4 v0.9.2 — Training Progress as State

Training-loop counters should become explicit state rather than local variables spread through examples.

Introduce an educational structure such as:

```python
TrainingState(
    epoch=...,
    global_step=...,
    batch_in_epoch=...,
    best_metric=...,
)
```

Core required fields:

```text
epoch
global_step
batch_in_epoch or resume-boundary policy
```

Optional educational metadata:

```text
best_metric
run_id
notes
```

Do not mix Tensor/model weights into this progress object; it represents **runtime position**, not all checkpoint payload.

## Resume-boundary decision

v0.9 must explicitly choose and document one of:

```text
epoch-boundary resume
```

or:

```text
mid-epoch exact resume
```

For a first educational implementation, epoch-boundary or controlled step-based deterministic datasets may be preferable. If mid-epoch resume is claimed, sampler/iterator state must genuinely be captured and tested.

---

# 3.5 v0.9.3 — Unified Checkpoint State Schema

Create one explicit logical checkpoint representation.

Conceptual payload:

```text
CheckpointState
├── schema_version
├── framework_version
├── model_state
├── optimizer_state
├── scheduler_state
├── rng_state
├── training_state
└── metadata
```

## Why schema version matters

A checkpoint is a data contract.

Future releases may change optimizer/runtime representation. Therefore serialized data should contain a small schema version rather than assuming every future reader understands every historical layout.

## Requirements

- model state is collected through Module state APIs;
- optimizer state is collected through optimizer state APIs;
- scheduler state is collected through scheduler state APIs;
- runtime progress is explicit;
- RNG state is explicit;
- schema/framework metadata are explicit;
- load validates required sections;
- unexpected/missing/incompatible state fails clearly when strict mode requires it.

Avoid storing live Python object references inside checkpoint state.

---

# 3.6 v0.9.4 — Inspectable Serialization Format

## Goal

The educational objective is to understand what is saved, not hide it inside one opaque blob.

Prefer an inspectable design such as:

```text
checkpoint/
├── metadata.json
└── tensors.npz
```

or a single documented NPZ container plus structured JSON metadata.

Suggested separation:

```text
NumPy arrays
→ NPZ

scalar/list/dict metadata
→ JSON-compatible structure
```

Avoid `pickle` as the primary mechanism unless a clear reason is documented.

## Requirements

- save;
- load;
- schema version validation;
- dtype/shape preservation;
- deterministic key mapping;
- readable metadata;
- clear malformed-file errors;
- no silent state omission.

### File-write safety

Where reasonable, demonstrate a small atomic-ish pattern:

```text
write temporary file
      ↓
fsync/close boundary where practical
      ↓
rename/replace target
```

Do not overbuild a production storage system; explain the failure mode being avoided.

---

# 3.7 v0.9.5 — Restore Semantics

Loading bytes is not equivalent to restoring training.

Implement one high-level restore path that applies state in a deliberate order.

Conceptually:

```text
load checkpoint data
      ↓
validate schema
      ↓
restore model
      ↓
restore optimizer
      ↓
restore scheduler
      ↓
restore RNG
      ↓
restore progress
```

The exact ordering should be documented and tests should prove the chosen semantics.

## Device rule

Checkpoint tensors should have an explicit device policy.

For v0.9 a simple educational rule is acceptable, for example:

```text
serialized arrays are device-neutral NumPy values
load restores into the already-configured target Module/Parameter devices
```

or an explicit `map_location`-style concept if useful.

Do not accidentally convert EDU parameters to CPU merely because NPZ stores host arrays.

---

# 3.8 v0.9.6 — Deterministic Resume Experiment

This is the central integration test of the version.

Run A — uninterrupted:

```text
initialize with seed S
      ↓
train steps 1 ... N
      ↓
record losses / params / optimizer / scheduler state
```

Run B — interrupted:

```text
initialize with same seed S
      ↓
train steps 1 ... K
      ↓
save checkpoint
      ↓
construct fresh objects
      ↓
load checkpoint
      ↓
train K+1 ... N
```

Then assert equivalence at the appropriate precision:

```text
loss trajectory after resume
model parameters
optimizer moments/momentum
scheduler LR
training counters
RNG continuation
```

The test should fail if any important state category is intentionally removed, making the lesson concrete.

Prefer a tiny deterministic MLP/dataset so CI remains fast.

---

# 3.9 v0.9.7 — Failure / Compatibility Semantics

Checkpoint systems are valuable partly because they reject invalid state clearly.

Tests should include:

- missing model key;
- wrong parameter shape;
- incompatible optimizer parameter-group structure;
- missing scheduler state when expected;
- unknown schema version;
- corrupted/missing metadata;
- malformed NPZ archive;
- device restoration policy;
- strict vs explicitly non-strict behavior where supported.

Never silently reshape parameters to make a checkpoint load.

---

# 3.10 Suggested source layout

Possible new package:

```text
mini_pytorch/runtime/
├── __init__.py
├── rng.py
├── state.py
└── checkpoint.py
```

Responsibilities:

```text
rng.py
→ framework RNG state / seed utilities

state.py
→ TrainingState / logical CheckpointState

checkpoint.py
→ serialization / deserialization / restore orchestration
```

Keep persistence code out of:

```text
Tensor
AutogradEngine
backend kernels
Dispatcher
```

Existing Module/Optimizer/Scheduler state APIs should be reused and extended only when a genuine missing primitive is discovered.

---

# 3.11 v0.9 tests

Add:

```text
tests/test_runtime_checkpoint.py
```

Minimum coverage:

### RNG

- same seed gives same sequence;
- save/restore RNG state continues identical sequence;
- checkpoint round-trip preserves RNG continuation.

### TrainingState

- defaults/validation;
- epoch/global-step round-trip;
- invalid counters rejected.

### Serialization

- state round-trip;
- array dtype/shape preservation;
- metadata readable;
- malformed checkpoint errors;
- unknown schema error.

### Framework restore

- Module state restoration;
- optimizer state restoration;
- scheduler state restoration;
- target device preserved.

### Integration

- uninterrupted training vs interrupted+resume equivalence;
- LR sequence identical after resume;
- optimizer state identical;
- future random draws identical.

Full historical suite must remain green.

---

# 3.12 v0.9 educational example

Add:

```text
examples/12_checkpoint_resume.py
```

It should show:

```text
seed
 ↓
train a few steps
 ↓
save checkpoint
 ↓
print/inspect checkpoint contents
 ↓
create fresh model/optimizer/scheduler
 ↓
load
 ↓
resume
 ↓
compare with uninterrupted reference
```

Keep the example small enough to run quickly without downloading MNIST.

---

# 3.13 v0.9 documentation

Add:

```text
docs/TRAINING_RUNTIME.md
```

It should explain:

- parameter state vs optimizer state;
- scheduler state;
- RNG seed vs RNG state;
- training progress state;
- checkpoint schema;
- serialization vs restoration;
- resume boundary;
- deterministic-resume experiment;
- why model weights alone are insufficient.

Version branch must also update:

```text
README.md
version.md
Agents.md
```

---

# 3.14 Definition of Done for v0.9

Before freeze, the version must demonstrate:

- explicit framework RNG state capture/restore;
- explicit TrainingState;
- schema-versioned logical checkpoint state;
- inspectable serialization format;
- model/optimizer/scheduler/RNG/progress round-trip;
- clear strict validation failures;
- documented device restoration policy;
- deterministic uninterrupted-vs-resumed training equivalence;
- no persistence logic embedded into Tensor math/Autograd Engine;
- educational docs/example complete;
- all v0.8 and earlier regression tests green;
- exact final `v0.9` branch-head GitHub Actions green;
- frozen tree integrated into `main`.

## Explicit non-goals for v0.9

Do not add yet:

- distributed checkpointing;
- sharded optimizer state;
- remote/object-store checkpoint service;
- async checkpoint writer;
- full DataLoader worker-state restoration;
- NCCL/DDP;
- compiler graph serialization;
- production security/cryptographic signing.

---

# 4. Future milestones

## v0.10 — Distributed / DDP Concepts

Core question:

> If every process has a model replica, when and how are gradients synchronized so replicas remain equivalent?

Planned concepts:

- DistributedSampler;
- ProcessGroup abstraction;
- simulated/CPU AllReduce;
- DDP wrapper;
- Autograd gradient-ready hooks;
- bucket construction;
- synchronization ordering;
- world size / rank semantics.

Architecture target:

```text
Autograd Engine
      ↓
gradient-ready hook
      ↓
DDP reducer
      ↓
grad buckets
      ↓
ProcessGroup.all_reduce
      ↓
replica-consistent grads
```

Goal: architecture, not reimplementing NCCL.

## v0.11 — Graph IR and Eager-to-Graph Capture

Core concepts:

- Node / Value / Graph;
- supported-op tracing;
- graph interpreter;
- explicit graph breaks;
- eager-vs-graph equivalence.

## v0.12 — Dynamo-like Guards and Graph Cache

Core concepts:

- shape/dtype/device/constant guards;
- graph cache;
- guard failure;
- recompilation;
- basic dynamic-shape experiments.

## v0.13 — AOTAutograd-like Forward/Backward Capture

Core concepts:

- captured forward graph;
- captured backward graph;
- saved intermediates contract;
- eager vs captured gradient equivalence;
- preparation for true differentiable/lowered backward graphs.

## v0.14 — Inductor-like Lowering and Kernel Fusion

Core concepts:

- lower-level loop IR;
- pointwise fusion;
- reduction lowering;
- generated Python/NumPy kernels;
- eager vs compiled correctness/performance comparison.

## v0.15 — Runtime / Performance Mechanics

Potential concepts after the compiler path is understandable:

- allocator/caching concepts;
- scheduling;
- prefetching;
- streams/events as architecture;
- kernel launch/runtime overhead;
- profiling.

## v1.0 — Integrated Architecture

The goal is not feature parity with PyTorch. v1.0 should be reached when the repository can coherently explain and demonstrate the main dependency chain:

```text
Tensor / Storage
      ↓
Autograd
      ↓
Module / Optimizer
      ↓
Data pipeline
      ↓
Dispatcher / Backend
      ↓
NN operators
      ↓
Autograd Engine
      ↓
Training runtime
      ↓
Distributed concepts
      ↓
Graph capture
      ↓
AOT backward
      ↓
Lowering / fusion
```

---

# 5. Version-development protocol

For every future version:

```text
read main/Agents.md + Plan.md
        ↓
create version branch from latest main
        ↓
write branch version.md + Agents.md first
        ↓
implement smallest readable reference mechanism
        ↓
add invariant + error + numerical/integration tests
        ↓
add educational example + architecture docs
        ↓
run full regression
        ↓
exact final branch-head CI green
        ↓
freeze branch
        ↓
integrate validated state into main
        ↓
restore canonical main docs
        ↓
advance this Plan.md
```

---

# 6. Current directive

`v0.8` is complete and frozen. `main` contains the validated **Advanced Autograd Engine** milestone with a **123-test** regression baseline.

The next branch should be **`v0.9`**, created from current `main`, and should focus on **Training Runtime / Serialization / Reproducibility**.

The central experiment of v0.9 must be:

```text
uninterrupted training
       ==
checkpoint → fresh process-like objects → restore → resume
```

for model parameters, optimizer dynamics, scheduler state, RNG continuation and training progress—not merely similar final loss.