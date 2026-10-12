# mini-pytorch v0.9

## 1. Version identity

- **Branch:** `v0.9`
- **Previous version:** `v0.8`
- **Theme:** Training Runtime / Serialization / Reproducibility
- **Status:** active development
- **Target package version:** `0.9.0`
- **Starting regression baseline:** `123/123 tests passed`

v0.9 answers:

> What exact state must be captured, validated and restored so a stopped training run can continue with the same future behavior?

## 2. Core difference from v0.8

v0.8 made backward execution explicit:

```text
forward graph → AutogradEngine → optimizer step
```

v0.9 makes the training process itself explicit state:

```text
Module state
Optimizer state
Scheduler state
RNG state
Training progress
      ↓
CheckpointState
      ↓
validated serialization
      ↓
restore into fresh objects
      ↓
deterministic continuation
```

The milestone is not `pickle.dump(model)`. It is an inspectable training-state contract.

## 3. Planned mechanisms

### 3.1 Framework RNG

Add a small NumPy-first RNG layer:

```python
manual_seed(42)
get_rng_state()
set_rng_state(state)
```

A seed initializes a stream; a checkpoint restores the **current RNG stream state**.

Framework-owned parameter initialization and unseeded stochastic operations should use this stream where practical.

### 3.2 TrainingState

Introduce explicit runtime progress:

```text
epoch
global_step
batch_in_epoch
best_metric
optional metadata
```

v0.9 will guarantee deterministic resume at an explicitly documented boundary rather than pretend every arbitrary DataLoader iterator is resumable.

### 3.3 CheckpointState schema

Logical state:

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

No live Parameter/Module objects belong in the serialized payload.

### 3.4 Inspectable serialization

Use a transparent checkpoint directory:

```text
checkpoint/
├── metadata.json
└── tensors.npz
```

NumPy arrays live in NPZ. JSON stores structure, array references, schema/framework versions and runtime metadata. `pickle` is not the primary format.

### 3.5 Restore orchestration

Loading bytes and restoring training are separate steps.

Restore order:

```text
validate schema
   ↓
model
   ↓
optimizer
   ↓
scheduler
   ↓
training progress
   ↓
RNG last
```

Restoring RNG last prevents construction of fresh model/optimizer objects from disturbing future random choices.

Serialized arrays are device-neutral host values. Loading into a model already placed on CPU/EDU preserves that target placement.

### 3.6 Deterministic resume experiment

Central integration proof:

```text
Run A: seed S → train 1..N
Run B: seed S → train 1..K → save → fresh objects → restore → train K+1..N
```

Compare:

- loss continuation;
- model parameters;
- optimizer moments/state;
- scheduler LR/state;
- TrainingState counters;
- future framework RNG draws.

## 4. Planned source layout

```text
mini_pytorch/runtime/
├── __init__.py
├── rng.py
├── state.py
└── checkpoint.py

tests/test_runtime_checkpoint.py
docs/TRAINING_RUNTIME.md
examples/12_checkpoint_resume.py
```

Existing Module/Optimizer/Scheduler state APIs should be reused and only strengthened where restore validation needs a missing primitive.

## 5. Compatibility invariants

v0.9 must preserve:

- v0.4 Storage/view/version safety;
- v0.5 Dispatcher behavior;
- v0.7 CPU/EDU placement;
- v0.8 Autograd graph lifecycle/hooks;
- leaf gradient semantics;
- existing Optimizer/Scheduler APIs;
- Dataset/DataLoader/CNN regressions.

Persistence logic must stay out of Tensor numerical kernels and AutogradEngine traversal.

## 6. Required tests

At minimum:

- same framework seed → same random sequence;
- RNG save/restore → identical continuation;
- TrainingState validation/round-trip;
- checkpoint model/optimizer/scheduler round-trip;
- target device preserved on restore;
- metadata JSON + tensor NPZ round-trip;
- unknown schema rejected;
- missing required section rejected;
- wrong model shape rejected;
- optimizer parameter-group mismatch rejected;
- missing scheduler state rejected when scheduler restore is requested;
- deterministic uninterrupted vs resumed training equivalence;
- future RNG draws identical after resume;
- complete historical suite remains green.

## 7. Explicit non-goals

Do not add in v0.9:

- distributed/sharded checkpoints;
- remote object-store persistence;
- async checkpoint writers;
- exact arbitrary mid-epoch DataLoader worker restoration;
- DDP/NCCL;
- compiler graph serialization;
- cryptographic signing/encryption.

## 8. Definition of Done

Before freezing v0.9:

- [ ] framework RNG state capture/restore implemented
- [ ] TrainingState implemented
- [ ] schema-versioned CheckpointState implemented
- [ ] inspectable JSON + NPZ serialization implemented
- [ ] model/optimizer/scheduler/RNG/progress restore implemented
- [ ] strict compatibility failures tested
- [ ] device restoration rule tested
- [ ] deterministic resume experiment passes
- [ ] README/docs/example updated
- [ ] package version is `0.9.0`
- [ ] all previous regression tests pass
- [ ] exact final v0.9 branch-head CI is green
- [ ] frozen validated tree is integrated into `main`

## 9. Next version

After v0.9, `v0.10` should focus on **Distributed / DDP Concepts**:

```text
Autograd gradient-ready hook
        ↓
DDP reducer / buckets
        ↓
ProcessGroup.all_reduce
        ↓
replica-consistent gradients
```
