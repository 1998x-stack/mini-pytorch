# Agents.md — mini-pytorch v0.9 Frozen Branch Contract

> Read this file before modifying the frozen `v0.9` branch.

## Branch identity

- Branch: `v0.9`
- Previous version: `v0.8`
- Primary mental model: **Training Runtime / Serialization / Reproducibility**
- Status: **frozen historical learning snapshot**
- Package version: `0.9.0`
- Validated baseline: `140/140 tests passed`

## 1. What v0.9 teaches

```text
Module / Optimizer / Scheduler
            +
         RNG state
            +
      TrainingState
            ↓
      CheckpointState
            ↓
metadata.json + tensors.npz
            ↓
     validated restore
            ↓
deterministic continuation
```

Frozen v0.9 includes:

- framework RNG seed/state APIs;
- framework-owned RNG integration for Linear/Conv2d/default Dropout/default RandomSampler;
- explicit `TrainingState`;
- schema-versioned `CheckpointState`;
- inspectable JSON + NPZ serialization;
- staged checkpoint publication;
- model/optimizer/scheduler restore;
- RNG restore-last semantics;
- target-device preservation;
- strict compatibility failures;
- exact deterministic-resume integration proof.

## 2. Historical branch rule

This branch is frozen.

Allowed changes only:

- documentation corrections;
- tests clarifying promised v0.9 behavior;
- narrowly scoped correctness fixes inside the v0.9 runtime/checkpoint boundary.

Forbidden:

- v0.10 DDP / ProcessGroup / AllReduce work;
- distributed/sharded checkpointing;
- async/remote checkpoint infrastructure;
- compiler serialization;
- unrelated NN/operator expansion.

## 3. Frozen architecture boundary

Persistence remains above numerical execution:

```text
Tensor / Storage / Dispatcher / AutogradEngine
                 ↑
           no persistence code

Module / Optimizer / Scheduler
                 ↓
             state_dict
                 ↓
          runtime/checkpoint
```

Do not move checkpoint behavior into Tensor, kernels, Dispatcher or AutogradEngine.

## 4. Frozen RNG semantics

```text
manual_seed(S)
→ restart framework stream

get_rng_state()
→ snapshot current stream position

set_rng_state(state)
→ resume saved stream position
```

Framework-owned RNG covers:

- Linear initialization;
- Conv2d initialization;
- Dropout with `seed=None`;
- RandomSampler with `seed=None`.

Explicitly seeded Dropout/Sampler instances retain private NumPy generators. Frozen v0.9 does not claim their arbitrary partial iterator/private RNG position is automatically included in the global checkpoint.

## 5. Frozen TrainingState semantics

Required runtime counters:

```text
epoch >= 0
global_step >= 0
batch_in_epoch >= 0
```

Optional small metadata:

```text
best_metric
run_id
notes
```

TrainingState is loop position, not model/optimizer tensor state.

## 6. Frozen checkpoint schema

Schema version:

```text
1
```

Required sections:

```text
framework_version
model_state
optimizer_state
scheduler_state
rng_state
training_state
metadata
```

Unknown schema versions must fail clearly rather than be interpreted heuristically.

## 7. Frozen serialization format

```text
checkpoint/
├── metadata.json
└── tensors.npz
```

Rules:

- top-level schema/framework/training fields remain human-readable;
- ndarrays live in NPZ;
- nested dict/list/tuple structure is explicit in JSON;
- integer optimizer state keys preserve their type;
- pickle is not the primary format;
- dtype and shape are preserved.

## 8. Frozen restore order

```text
validate
  ↓
model
  ↓
optimizer
  ↓
scheduler
  ↓
TrainingState
  ↓
RNG LAST
```

RNG restore-last is an invariant because fresh object construction may consume framework randomness.

## 9. Device invariant

Serialized arrays are device-neutral values.

Restoring state into existing Parameter/Buffer objects must preserve the target object's configured device. Do not infer CPU placement merely because checkpoint storage uses NumPy arrays.

## 10. Strict failure semantics

Do not weaken the frozen errors for:

- unknown schema;
- missing/corrupt checkpoint component files;
- model shape mismatch;
- optimizer parameter-group mismatch;
- missing target scheduler for checkpoint scheduler state in strict mode;
- overwrite protection.

Do not silently reshape or drop required state.

## 11. Deterministic-resume invariant

The release proof is:

```text
Run A: seed → train 1..N

Run B: seed → train 1..K
              ↓
             save
              ↓
        fresh objects
              ↓
             load
              ↓
          train K+1..N
```

Validated equality includes:

- losses;
- model arrays;
- Adam state;
- StepLR state/current LR;
- TrainingState;
- future framework RNG draws.

Do not describe v0.9 as deterministic resume if a future fix breaks this integration test.

## 12. Resume-boundary invariant

v0.9 supports controlled step/epoch-boundary resume for framework-owned state.

It does not claim exact restoration of arbitrary partially-consumed multi-worker DataLoader iterators, worker RNGs, external libraries or explicitly private component RNGs.

## 13. Validation baseline

Validated release-candidate implementation:

```text
45e09013dfecfb86055220c1e3be41928cf7bc7d
```

GitHub Actions:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
140 tests
OK
```

Any future narrowly scoped v0.9 fix must rerun the complete suite and pass GitHub Actions on the exact modified branch HEAD.

## 14. Documentation map

```text
README.md
version.md
mini_pytorch/runtime/rng.py
mini_pytorch/runtime/state.py
mini_pytorch/runtime/checkpoint.py
tests/test_runtime_checkpoint.py
docs/TRAINING_RUNTIME.md
examples/12_checkpoint_resume.py
```

## 15. Next-version boundary

Distributed synchronization belongs to `v0.10`, created from the latest validated `main`, not from this frozen historical branch.
