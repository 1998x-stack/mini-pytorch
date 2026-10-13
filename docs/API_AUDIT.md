# v1.0 Public API and Naming Audit

> Scope: exported learning APIs on branch `v1.0`.
>
> Goal: make the integrated release internally consistent without erasing useful subsystem boundaries or chasing PyTorch feature parity.

## 1. Audit result

The v1.0 audit found **no reason for a breaking public-API cleanup**. The existing module boundaries are useful teaching surfaces and are already exercised by historical examples/tests.

The v1.0 changes are therefore deliberately small:

1. package version is centralized in `mini_pytorch/_version.py`;
2. `mini_pytorch.__version__` is exported explicitly;
3. checkpoint `framework_version` now defaults to that same package version;
4. compiler package documentation is updated from the v0.15 wording to the integrated v1.0 wording;
5. canonical documentation is rewritten to describe the actual complete API surface.

No historical public class/function is removed solely for cosmetic consistency.

---

## 2. Top-level `mini_pytorch`

Current public surface includes:

```text
Tensor
Storage
Device
Backend / backend_for / list_backends
Context / Function / GradFn / RemovableHandle
AutogradEngine / engine / no_grad
Dispatcher-facing types
functional
nn / optim / data / runtime / distributed / compiler
runtime checkpoint/RNG conveniences
__version__
```

### Decision

Keep the low-level classes public.

In a production high-level library, some of these might be hidden. In this project they are intentionally inspectable learning concepts.

### v1.0 cleanup

Before v1.0, version metadata was duplicated:

```text
mini_pytorch.__version__ = 0.15.0
checkpoint FRAMEWORK_VERSION = 0.9.0
pyproject version = 0.15.0
```

v1.0 centralizes runtime code on:

```text
mini_pytorch/_version.py
__version__ = 1.0.0
```

and package/checkpoint code consume that value. `pyproject.toml` uses the same release number as package metadata.

---

## 3. `mini_pytorch.nn`

Exports:

```text
Module
Parameter
Linear
ReLU / Sigmoid
MSELoss / CrossEntropyLoss
Sequential / Flatten
Conv2d / MaxPool2d
Dropout
BatchNorm1d / BatchNorm2d
```

### Decision

No rename/removal.

The names correspond cleanly to the educational mechanisms implemented in the package. Adding more layers only for API breadth is outside v1.0 scope.

---

## 4. `mini_pytorch.optim`

Exports:

```text
Optimizer
SGD
Adam / AdamW / Adamax
RMSprop
Adagrad
LRScheduler
LambdaLR / StepLR / MultiStepLR
ExponentialLR / CosineAnnealingLR / LinearLR
ReduceLROnPlateau
```

### Decision

No breaking changes.

The separation between optimizer algorithms and schedulers remains useful and matches the project's version history.

---

## 5. `mini_pytorch.data`

Exports:

```text
Dataset
TensorDataset
Sampler
SequentialSampler
RandomSampler
BatchSampler
DataLoader
default_collate
MNIST
```

### Decision

Keep `default_collate` public in the data namespace because collation is an explicit DataLoader learning boundary.

Do not add multiprocessing, worker, prefetch or pinned-memory APIs in v1.0.

---

## 6. `mini_pytorch.runtime`

Exports two categories.

### Framework RNG

```text
manual_seed
get_rng_state
set_rng_state
random / uniform / integers / permutation
```

### Checkpoint/progress

```text
TrainingState
CheckpointState
CHECKPOINT_SCHEMA_VERSION
CheckpointError
collect_checkpoint_state
save_checkpoint_state / load_checkpoint_state
restore_checkpoint_state
save_checkpoint / load_checkpoint
```

### Decision

Keep random sampling helpers inside `mini_pytorch.runtime` rather than promoting every helper to top-level. The top-level package exposes only seed/state control plus checkpoint conveniences.

This preserves a useful distinction:

```text
common user workflow
vs
runtime subsystem implementation surface
```

---

## 7. `mini_pytorch.distributed`

Exports:

```text
ProcessGroup / CollectiveWork
DistributedSampler
Reducer / GradientBucket / BucketEntry
DistributedDataParallel
broadcast_module_state
synchronize_ddp
error types
```

### Decision

No attempt to mimic the complete `torch.distributed` namespace.

These are intentionally explicit simulation objects so a reader can inspect rank/world-size, bucket and synchronization responsibilities.

---

## 8. `mini_pytorch.compiler`

This namespace is intentionally broad because the compiler sequence is a learning stack, not a single black-box `compile()` entry point.

Public categories:

```text
Graph IR
capture
GraphInterpreter
signatures / guards / cache
CompiledFunction / compile_graph
AOT differentiation
analysis / transforms
lowering / fusion
benchmark / profiling
lifetime analysis
MemoryPlan / ExecutionPlan
PlannedExecutor
```

### Decision

Keep these concepts public in v1.0.

Hiding them behind only `compile_graph()` would make the final project less useful for learning.

### Naming note

`compile_graph()` is an educational project API. It must not be described as a drop-in equivalent of `torch.compile()`.

`ExecutionPlan` is also project-specific; the mapping document explicitly avoids claiming a one-to-one public PyTorch class.

---

## 9. Error naming audit

Subsystem-specific errors remain preferable to one generic framework error:

```text
DispatchError
GraphBreakError
AOTDifferentiationError
LoweringError
RuntimeAnalysisError
ExecutionPlanningError
CheckpointError
DistributedError / ReducerError / CollectiveError
```

### Decision

No consolidation into a single exception hierarchy in v1.0.

The current names make the failure boundary visible, which is more valuable for this project than exception-tree uniformity.

---

## 10. Compatibility policy for v1.0

v1.0 should be read as the first integrated learning release, not as a promise of future semantic version compatibility with PyTorch.

Within this repository, however, the release follows these rules:

- no historical public name is removed without a clear architectural reason;
- examples use supported public module paths where available;
- private helpers remain private even if importing them could shorten an example;
- project-specific compiler names are documented as project-specific;
- package/checkpoint release metadata agree on `1.0.0`;
- future expansion after v1.0 requires a new explicit roadmap.

---

## 11. Audit conclusion

The v1.0 API work is intentionally conservative:

```text
keep readable boundaries
      +
fix release metadata drift
      +
document what is public and why
      -
no cosmetic breaking rewrite
```

That is consistent with the v1.0 goal: integrate the architecture instead of starting a new feature cycle.
