# Agents.md — mini-pytorch v0.9 Branch Contract

> Mandatory reading before changing the `v0.9` branch.

## Branch identity

- Branch: `v0.9`
- Previous version: `v0.8`
- Primary mental model: **Training Runtime / Serialization / Reproducibility**
- Status: active development
- Starting validated baseline: `123/123 tests passed`

## 1. Central question

> What exact state must be captured, validated and restored so a stopped training run can continue with the same future behavior?

The version is about training-state architecture, not opaque object dumping.

## 2. Allowed scope

```text
model state
optimizer state
scheduler state
RNG state
training progress
schema/version metadata
       ↓
CheckpointState
       ↓
serializer / loader
       ↓
deterministic resume
```

Allowed:

- framework RNG utilities;
- migration of framework-owned random initialization/stochastic behavior to that RNG where safe;
- explicit `TrainingState`;
- logical `CheckpointState`;
- JSON + NPZ serialization;
- strict validation and restore orchestration;
- small state-validation improvements to Module/Optimizer/Scheduler if required;
- deterministic resume tests/example/docs.

## 3. Forbidden scope

Do not add:

- DDP / ProcessGroup / AllReduce;
- distributed or sharded checkpoints;
- remote storage backends;
- asynchronous checkpoint workers;
- full arbitrary mid-epoch DataLoader worker restoration;
- real CUDA checkpoint runtime;
- compiler graph serialization;
- pickle as the primary educational format;
- unrelated NN/operator expansion.

## 4. Architecture boundary

Persistence belongs above numerical execution:

```text
Tensor / Storage / Dispatcher / AutogradEngine
                 ↑
           no checkpoint code

Module / Optimizer / Scheduler
                 ↓
             state_dict
                 ↓
             runtime/*
```

Do not embed serialization logic in Tensor, kernels, Dispatcher or AutogradEngine.

## 5. RNG contract

The framework RNG must make this distinction explicit:

```text
manual_seed(S)
→ starts a stream

get_rng_state()
→ captures current position

set_rng_state(state)
→ resumes current position
```

Restoring only a seed is not a valid checkpoint-resume implementation.

Framework-owned randomness should use the framework RNG where practical. Explicit component-private RNGs may retain their own semantics, but deterministic-resume claims must clearly state which sources are captured.

## 6. TrainingState contract

Training progress is runtime position, not model state.

Required fields:

```text
epoch >= 0
global_step >= 0
batch_in_epoch >= 0
```

Optional metadata may include `best_metric`, `run_id`, and notes.

v0.9 should document an exact resume boundary. Do not claim arbitrary DataLoader iterator restoration unless sampler/worker position is genuinely captured and tested.

## 7. Checkpoint schema contract

Required logical sections:

```text
schema_version
framework_version
model_state
optimizer_state
scheduler_state
rng_state
training_state
metadata
```

Checkpoint data must not contain live Module/Parameter/Optimizer references.

Strict loading must reject incompatible or missing state instead of silently dropping or reshaping it.

## 8. Serialization contract

Preferred format:

```text
checkpoint/
├── metadata.json
└── tensors.npz
```

Rules:

- ndarray values are stored losslessly in NPZ;
- structure and scalar metadata remain JSON-readable;
- array references are deterministic;
- schema version is validated before restoration;
- malformed/missing files fail clearly;
- writes should use a temporary staging location before publishing the checkpoint directory.

## 9. Restore contract

Use a deliberate order:

```text
parse + validate
      ↓
model.load_state_dict
      ↓
optimizer.load_state_dict
      ↓
scheduler.load_state_dict
      ↓
TrainingState
      ↓
set_rng_state LAST
```

RNG is restored last so construction of fresh objects does not change the future stream after resume.

Serialized tensors are device-neutral NumPy values. Restoring into an already-configured CPU/EDU model must preserve that target device.

## 10. Deterministic-resume proof

Do not call v0.9 complete until CI proves:

```text
uninterrupted run
==
interrupted → save → fresh objects → load → continue
```

Compare at least:

- post-resume losses;
- model arrays;
- optimizer state arrays/counters;
- scheduler LR/state;
- training counters;
- next framework RNG draws.

Use a tiny deterministic dataset/model so CI stays fast.

## 11. Testing contract

Full regression:

```bash
python -m unittest discover -s tests -v
```

New tests should cover:

- RNG reproducibility/state restore;
- TrainingState validation;
- generic nested checkpoint serialization;
- model/optimizer/scheduler restore;
- target device preservation;
- schema errors;
- missing/corrupt files;
- optimizer group mismatch;
- scheduler presence mismatch;
- deterministic resume.

Exact final v0.9 branch-head GitHub Actions must be green before freeze.

## 12. Documentation contract

Before freeze update:

```text
README.md
version.md
Agents.md
docs/TRAINING_RUNTIME.md
examples/12_checkpoint_resume.py
mini_pytorch/__init__.py
pyproject.toml
```

## 13. Freeze rule

When exact final branch HEAD is green:

1. mark `version.md` frozen;
2. mark this `Agents.md` frozen;
3. do not modify framework code again;
4. fast-forward validated v0.9 into `main`;
5. restore canonical main README/Agents/Plan;
6. advance roadmap to `v0.10 Distributed / DDP Concepts`.
