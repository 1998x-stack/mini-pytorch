# mini-pytorch v0.9

## 1. Version identity

- **Branch:** `v0.9`
- **Previous version:** `v0.8`
- **Theme:** Training Runtime / Serialization / Reproducibility
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.9.0`
- **Validated regression baseline:** **140/140 tests passed**

v0.9 answers:

> What exact state must be captured, validated and restored so a stopped training run can continue with the same future behavior?

## 2. Core difference from v0.8

v0.8 made backward execution explicit:

```text
Tensor / GradFn / Context
        ↓
AutogradEngine
        ↓
Optimizer.step()
```

v0.9 makes the complete training process explicit state:

```text
Module state
Optimizer state
Scheduler state
RNG state
Training progress
      ↓
CheckpointState
      ↓
metadata.json + tensors.npz
      ↓
fresh objects
      ↓
validated restore
      ↓
deterministic continuation
```

The milestone is training-state architecture, not opaque object dumping.

## 3. Implemented mechanisms

### Framework RNG

Added:

```python
manual_seed(seed)
get_rng_state()
set_rng_state(state)
```

The framework uses one checkpointable NumPy PCG64 stream for framework-owned randomness.

Integrated with:

- `Linear` initialization;
- `Conv2d` initialization;
- `Dropout(seed=None)`;
- `RandomSampler(seed=None)`.

Explicitly seeded Dropout/Sampler instances preserve their historical private-generator semantics and are outside automatic framework-RNG checkpoint capture.

### TrainingState

Added explicit runtime progress:

```text
epoch
global_step
batch_in_epoch
best_metric
run_id
notes
```

Counters validate non-negative integer semantics.

### CheckpointState

Logical checkpoint representation:

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

Checkpoint collection is separate from disk serialization and object restoration.

### Inspectable serialization

Checkpoint layout:

```text
checkpoint/
├── metadata.json
└── tensors.npz
```

`metadata.json` exposes human-readable top-level format/schema/framework/training fields.
Nested structures use explicit tagged encoding; ndarray payloads live losslessly in `tensors.npz`.

Supported structure includes:

- dictionaries with string/integer/simple scalar keys;
- lists;
- tuples;
- NumPy scalar values;
- ndarrays with dtype/shape preservation.

This is sufficient to preserve optimizer state with positional integer parameter indices and nested moment arrays without pickle.

### Staged publication

Checkpoint writes use a temporary directory, fsync component files, then publish by rename. Existing targets are moved to a temporary backup during overwrite and restored if publication fails.

This is an educational atomic-ish pattern, not a distributed storage protocol.

### Restore orchestration

Restore order:

```text
validate checkpoint
      ↓
model.load_state_dict
      ↓
optimizer.load_state_dict
      ↓
scheduler.load_state_dict
      ↓
TrainingState
      ↓
set_rng_state(...) LAST
```

RNG is restored last because construction of fresh model objects may consume random values.

### Device-neutral checkpoint values

Serialized ndarray values are device-neutral host values.

Restoring into an already configured target Module writes into existing Parameter/Buffer Storage, so target placement is preserved. Tests prove CPU-saved model values can be restored into an EDU model without moving parameters back to CPU.

### Strict compatibility failures

v0.9 explicitly rejects:

- unknown checkpoint schema;
- missing metadata/tensor files;
- malformed NPZ archives;
- model parameter shape mismatch;
- incompatible optimizer parameter-group structure;
- checkpoint scheduler state without a target scheduler in strict mode;
- overwrite when `overwrite=False`.

No silent parameter reshape or required-state dropping is performed.

## 4. Deterministic resume proof

The central integration test uses:

```text
Linear
  ↓
ReLU
  ↓
Dropout(seed=None)
  ↓
Linear

Adam + StepLR + framework RNG
```

Run A:

```text
manual_seed(S)
→ train steps 1 ... N
→ record final state + future RNG draws
```

Run B:

```text
manual_seed(S)
→ train steps 1 ... K
→ save checkpoint
→ construct fresh model/optimizer/scheduler
→ restore checkpoint
→ train K+1 ... N
→ record final state + future RNG draws
```

CI proves exact equality of:

- full loss trajectory;
- model arrays;
- Adam state arrays/counters;
- StepLR state/current LR;
- TrainingState;
- future framework RNG values.

This is the version's definition of deterministic resume.

## 5. Resume boundary

v0.9 guarantees deterministic resume at controlled step/epoch boundaries using framework-owned RNG state.

`TrainingState.batch_in_epoch` can record position, but v0.9 intentionally does **not** claim restoration of an arbitrary partially-consumed multi-worker DataLoader iterator or external/private random sources.

## 6. New files

```text
mini_pytorch/runtime/__init__.py
mini_pytorch/runtime/rng.py
mini_pytorch/runtime/state.py
mini_pytorch/runtime/checkpoint.py

tests/test_runtime_checkpoint.py
docs/TRAINING_RUNTIME.md
examples/12_checkpoint_resume.py
```

Major modified files:

```text
mini_pytorch/nn/linear.py
mini_pytorch/nn/conv.py
mini_pytorch/nn/dropout.py
mini_pytorch/data/sampler.py
mini_pytorch/__init__.py
pyproject.toml
README.md
MANIFEST.txt
Agents.md
version.md
```

## 7. Preserved invariants

v0.9 preserves:

- Storage/stride/view/version-counter semantics;
- saved-version inplace safety;
- Dispatcher/redispatch behavior;
- CPU/EDU placement;
- AutogradEngine graph lifetime/hooks/retain-grad semantics;
- Module/Optimizer/Scheduler public state APIs;
- Dataset/DataLoader/CNN behavior;
- leaf gradient accumulation.

Checkpoint code does not live inside Tensor math, backend kernels, Dispatcher or AutogradEngine traversal.

## 8. Validation

Validated release-candidate implementation commit:

```text
45e09013dfecfb86055220c1e3be41928cf7bc7d
```

GitHub Actions environment:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

Result:

```text
Ran 140 tests in 0.990s
OK
```

This is the previous v0.8 baseline of 123 tests plus 17 dedicated v0.9 runtime/checkpoint tests.

## 9. v0.9-specific test coverage

### RNG

- manual seed reproducibility;
- RNG-state continuation restore;
- framework RNG parameter initialization;
- default Dropout RNG integration;
- default RandomSampler RNG integration.

### TrainingState

- round-trip;
- invalid counter errors.

### Serialization

- JSON + NPZ round-trip;
- top-level metadata inspection;
- optimizer nested arrays/tuple/integer-key preservation;
- unknown schema rejection;
- missing/corrupt file rejection;
- overwrite protection.

### Restore

- model state;
- optimizer state;
- scheduler state;
- training progress;
- target EDU device preservation;
- model shape mismatch;
- optimizer group mismatch;
- scheduler-presence strictness.

### Integration

- uninterrupted training vs save/reconstruct/load/continue exact equivalence;
- exact future RNG continuation.

## 10. Definition of Done

- [x] framework RNG capture/restore implemented
- [x] framework-owned stochastic mechanisms integrated
- [x] explicit `TrainingState` implemented
- [x] schema-versioned `CheckpointState` implemented
- [x] inspectable JSON + NPZ format implemented
- [x] staged checkpoint publication implemented
- [x] model/optimizer/scheduler/RNG/progress restore implemented
- [x] strict compatibility failures tested
- [x] target-device preservation tested
- [x] deterministic resume experiment passes
- [x] README/docs/example complete
- [x] package version is `0.9.0`
- [x] full historical regression passes
- [x] branch marked frozen pending exact frozen-head CI/integration

## 11. Recommended reading order

```text
mini_pytorch/runtime/rng.py
        ↓
mini_pytorch/runtime/state.py
        ↓
mini_pytorch/runtime/checkpoint.py
        ↓
mini_pytorch/nn/linear.py + conv.py + dropout.py
        ↓
mini_pytorch/data/sampler.py
        ↓
tests/test_runtime_checkpoint.py
        ↓
docs/TRAINING_RUNTIME.md
        ↓
examples/12_checkpoint_resume.py
```

## 12. Explicit non-goals

v0.9 intentionally does not implement:

- arbitrary mid-epoch multi-worker iterator restoration;
- distributed/sharded checkpointing;
- async checkpoint writers;
- object-store/remote persistence;
- DDP/NCCL state;
- compiler graph serialization;
- cryptographic checkpoint signing/encryption.

## 13. Learning goals

After v0.9 the reader should be able to explain:

1. why model weights are not the full training state;
2. seed vs current RNG stream state;
3. why optimizer and scheduler state affect future parameters;
4. why training progress belongs in an explicit runtime object;
5. state collection vs serialization vs restoration;
6. why a checkpoint needs a schema version;
7. why RNG should be restored after fresh object construction;
8. how device-neutral serialized values can preserve target device placement;
9. what deterministic resume actually needs to prove.

## 14. Next version

`v0.10` should focus on **Distributed / DDP Concepts**:

```text
Autograd gradient-ready signal
        ↓
DDP reducer
        ↓
gradient buckets
        ↓
ProcessGroup.all_reduce
        ↓
replica-consistent gradients
```
