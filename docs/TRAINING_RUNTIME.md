# Training Runtime / Serialization / Reproducibility

mini-pytorch v0.9 asks a deceptively simple question:

> If training stops now, what must be saved so the **future** run is the same after restart?

Saving model weights is only one part of the answer.

---

## 1. A training process is a state machine

A useful mental model is:

```text
                 training process
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
     Module         Optimizer        Scheduler
       │               │                │
       └───────────────┼────────────────┘
                       │
              runtime progress
                       +
                   RNG state
                       │
                       ▼
                CheckpointState
```

The important categories are:

```text
model parameters/buffers
optimizer moments/momentum + parameter-group options
scheduler progress + current LR
random stream position
training counters
schema/framework metadata
```

If any state that influences the next step is omitted, resume can diverge.

---

## 2. Seed is not RNG state

A seed chooses the beginning of a random stream:

```python
manual_seed(42)
```

Conceptually:

```text
seed 42
  ↓
r0, r1, r2, r3, r4, ...
```

After three draws the current state means:

```text
next draw = r3
```

Re-running `manual_seed(42)` goes back to `r0`; it does **not** resume from `r3`.

v0.9 therefore exposes:

```python
manual_seed(seed)
get_rng_state()
set_rng_state(state)
```

Checkpointing stores the current PCG64 state.

Framework-owned randomness includes:

- `Linear` initialization;
- `Conv2d` initialization;
- `Dropout(seed=None)`;
- `RandomSampler(seed=None)`.

An explicitly seeded Dropout/Sampler intentionally owns a private NumPy Generator for backwards-compatible component-local determinism. v0.9 does not claim arbitrary private iterator/RNG state is automatically checkpointed.

---

## 3. Why RNG is restored last

Consider restart:

```text
load checkpoint
      ↓
construct fresh model
      ↓
construct fresh optimizer/scheduler
```

Model construction itself may consume random values. If checkpoint RNG were restored **before** creating the fresh model, those constructor draws would move the stream and future Dropout masks would differ.

v0.9 restore order is therefore:

```text
validate checkpoint
      ↓
model.load_state_dict
      ↓
optimizer.load_state_dict
      ↓
scheduler.load_state_dict
      ↓
restore TrainingState
      ↓
set_rng_state(saved_state)   ← last
```

Fresh-object construction may consume any amount of randomness before load; the saved stream position still becomes authoritative at the end.

---

## 4. TrainingState is not model state

`TrainingState` stores runtime position:

```python
TrainingState(
    epoch=2,
    global_step=731,
    batch_in_epoch=19,
    best_metric=0.913,
)
```

It does not contain tensors or optimizer moments.

The separation is intentional:

```text
Module.state_dict()
→ learnable/persistent model state

Optimizer.state_dict()
→ optimizer dynamics

Scheduler.state_dict()
→ LR schedule position

TrainingState
→ loop position / small runtime metadata
```

v0.9 can represent `batch_in_epoch`, but its exact deterministic-resume demonstration uses a controlled step/epoch boundary. It does not claim restoration of multiprocessing workers or an arbitrary partially consumed DataLoader iterator.

---

## 5. Logical CheckpointState

Before any bytes are written, v0.9 builds an in-memory logical contract:

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

This separation matters:

```text
collect state
!=
serialize state
!=
restore state
```

You can inspect or validate `CheckpointState` without touching disk or mutating a model.

---

## 6. Why schema_version exists

A checkpoint is a data contract, not just a bag of bytes.

Today:

```text
schema_version = 1
```

Future versions may change representation. A loader must know whether it understands the data rather than silently guessing.

Unknown schema versions therefore fail clearly.

---

## 7. Inspectable JSON + NPZ format

A checkpoint path is a directory:

```text
checkpoint/
├── metadata.json
└── tensors.npz
```

### metadata.json

Top-level fields are immediately readable:

```json
{
  "format": "mini-pytorch-checkpoint",
  "schema_version": 1,
  "framework_version": "0.9.0",
  "training_state": {
    "epoch": 2,
    "global_step": 731,
    "batch_in_epoch": 19
  }
}
```

Complex nested structures are encoded explicitly. ndarray values become named references.

### tensors.npz

All ndarray payloads are stored losslessly in NumPy NPZ:

```text
array_000000
array_000001
array_000002
...
```

The JSON structure points to those deterministic names.

This preserves:

- dtype;
- shape;
- optimizer moment arrays;
- model parameters/buffers;
- tuples/lists;
- integer dictionary keys used by optimizer parameter indices.

No pickle is required.

---

## 8. Why optimizer state matters

Adam is not determined by model weights alone.

Its update depends on historical state:

```text
parameter
+ gradient
+ exp_avg
+ exp_avg_sq
+ step counter
+ parameter-group LR/options
→ next parameter value
```

If only model weights are restored:

```text
same model now
≠
same model after next optimizer step
```

v0.9 serializes and restores the existing optimizer `state_dict()` representation, whose parameter identity is based on stable positional indices rather than Python object ids.

---

## 9. Why scheduler state matters

A scheduler can make two otherwise identical optimizers take different steps.

For `StepLR`, checkpoint state includes:

```text
base_lrs
last_epoch
_step_count
_last_lr
```

Fresh scheduler construction happens first, then its saved state is restored. The scheduler also writes its restored current LR back into optimizer parameter groups.

A checkpoint containing scheduler state cannot be strictly restored without a target scheduler.

---

## 10. Device-neutral serialized arrays

NPZ stores host NumPy arrays, but that does not mean restore forces model parameters to CPU.

The rule is:

```text
checkpoint arrays = device-neutral values
existing target Module placement = authoritative device
```

Example:

```text
save CPU Linear
      ↓
construct fresh Linear.to("edu")
      ↓
load_state_dict writes values into existing EDU Storage
      ↓
parameters remain on EDU
```

This mirrors the conceptual purpose of a `map_location` policy without adding accelerator complexity in v0.9.

---

## 11. Deterministic resume experiment

The central v0.9 test uses a small stochastic model:

```text
Linear
  ↓
ReLU
  ↓
Dropout(seed=None)
  ↓
Linear
```

with:

```text
Adam
StepLR
framework RNG
```

### Run A

```text
manual_seed(S)
      ↓
construct objects
      ↓
train steps 1 ... N
      ↓
record final state and future random draws
```

### Run B

```text
manual_seed(S)
      ↓
construct objects
      ↓
train steps 1 ... K
      ↓
save checkpoint
      ↓
construct completely fresh objects
      ↓
load checkpoint
      ↓
train K+1 ... N
      ↓
record final state and future random draws
```

CI asserts exact equality of:

```text
loss trajectory
model arrays
Adam state arrays/counters
StepLR state/current LR
TrainingState
next framework RNG values
```

This is stronger than asserting that a checkpoint can merely be opened.

---

## 12. Serialization safety boundary

v0.9 writes into a temporary staging directory first:

```text
build complete temporary checkpoint
        ↓
close + fsync component files
        ↓
publish directory by rename
```

When overwriting, the old target is temporarily renamed to a backup and restored if publication fails.

This is an educational failure-safety pattern, not a production distributed checkpoint protocol.

---

## 13. Strict failures are part of the API

v0.9 tests clear failure for:

- unknown schema version;
- missing metadata/tensor files;
- malformed NPZ;
- wrong model parameter shape;
- incompatible optimizer parameter-group count;
- scheduler state without a target scheduler;
- overwrite disabled for an existing checkpoint.

The loader never silently reshapes parameters to make incompatible state fit.

---

## 14. Reading order

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
examples/12_checkpoint_resume.py
```

---

## 15. Explicit non-goals

v0.9 does not implement:

- arbitrary mid-epoch multi-worker iterator restoration;
- distributed/sharded checkpoints;
- remote/object-store writes;
- async checkpoint threads;
- DDP/NCCL state;
- compiler graph serialization;
- checkpoint encryption/signing.

The next architectural layer is distributed gradient synchronization, not a larger storage service.
