# Learning Path — v1.0

> Recommended order for learning the complete mini-pytorch architecture.
>
> The goal is not to memorize APIs. Each round should end with one mechanism you can explain without looking at the code.

## Round 1 — Dynamic Autograd MVP

Run:

```bash
python examples/01_autograd.py
```

Read:

```text
mini_pytorch/tensor.py
mini_pytorch/autograd.py
mini_pytorch/ops.py
```

Explain:

```text
forward numerical result
+
GradFn graph construction
+
reverse chain rule
```

Historical branch: `v0.1`.

---

## Round 2 — Optimizer state and schedulers

Run:

```bash
python examples/04_parameter_groups.py
python examples/05_lr_schedulers.py
```

Read:

```text
mini_pytorch/optim/optimizer.py
mini_pytorch/optim/sgd.py
mini_pytorch/optim/adam.py
mini_pytorch/optim/lr_scheduler.py
```

Explain why:

```text
Optimizer != Autograd
scheduler state != optimizer parameter state
parameter groups can carry different hyperparameters
```

Historical branch: `v0.2`.

---

## Round 3 — Dataset / Sampler / DataLoader

Read:

```text
mini_pytorch/data/dataset.py
mini_pytorch/data/sampler.py
mini_pytorch/data/dataloader.py
```

Then inspect:

```bash
python examples/06_mnist_mlp.py --help
```

The MNIST example may require external data, so source reading is sufficient when offline.

Explain:

```text
Dataset = what is sample i?
Sampler = which index comes next?
BatchSampler = which indices form one batch?
DataLoader = fetch + collate
```

Historical branch: `v0.3`.

---

## Round 4 — Storage / stride / views

Run:

```bash
python examples/07_tensor_memory.py
```

Read:

```text
mini_pytorch/storage.py
Tensor.view
Tensor.reshape
Tensor.contiguous
Tensor.transpose / permute / slicing
```

Explain:

```text
Tensor != allocation
view may share Storage
reshape may copy
in-place mutation bumps shared version
```

Historical branch: `v0.4`.

---

## Round 5 — Dispatcher

Run:

```bash
python examples/08_dispatcher.py
```

Read:

```text
mini_pytorch/dispatcher.py
mini_pytorch/operator_registry.py
mini_pytorch/kernels.py
```

Explain:

```text
schema
→ DispatchKeySet
→ policy/backend selection
→ kernel/fallback
→ redispatch
```

Historical branch: `v0.5`.

---

## Round 6 — CNN operator stack and backend placement

Run:

```bash
python examples/09_cnn_mnist.py --help
python examples/10_device_backend.py
```

Read:

```text
mini_pytorch/nn_ops.py
mini_pytorch/nn/conv.py
mini_pytorch/nn/pooling.py
mini_pytorch/nn/batchnorm.py
mini_pytorch/backend.py
mini_pytorch/device.py
```

Explain:

```text
reference numerical operator
→ dispatched backend execution
→ Module wrapper
```

Historical branches: `v0.6`, `v0.7`.

---

## Round 7 — Advanced Autograd Engine

Run:

```bash
python examples/11_autograd_engine.py
```

Read:

```text
mini_pytorch/autograd_engine.py
saved-tensor version logic in autograd.py
Tensor.retain_grad / hooks
```

Explain the difference between:

```text
leaf grad ownership
graph lifetime
saved tensor lifetime
retain_graph
retain_grad
hooks
```

Historical branch: `v0.8`.

---

## Round 8 — Training runtime and deterministic resume

Run:

```bash
python examples/12_checkpoint_resume.py
```

Read:

```text
mini_pytorch/runtime/rng.py
mini_pytorch/runtime/state.py
mini_pytorch/runtime/checkpoint.py
```

Explain why a true resumable checkpoint includes more than model weights:

```text
model
optimizer
scheduler
RNG
training progress
```

Historical branch: `v0.9`.

---

## Round 9 — Distributed / DDP concepts

Run:

```bash
python examples/13_ddp_simulation.py
```

Read:

```text
mini_pytorch/distributed/process_group.py
mini_pytorch/distributed/reducer.py
mini_pytorch/distributed/ddp.py
mini_pytorch/distributed/sampler.py
```

Explain:

```text
local backward
→ gradient readiness
→ bucket/reducer
→ collective synchronization
→ ordinary optimizer step
```

Historical branch: `v0.10`.

---

## Round 10 — Graph IR and guarded cache

Run:

```bash
python examples/14_graph_capture.py
python examples/15_graph_cache.py
```

Read:

```text
mini_pytorch/compiler/ir.py
mini_pytorch/compiler/capture.py
mini_pytorch/compiler/signature.py
mini_pytorch/compiler/guards.py
mini_pytorch/compiler/cache.py
mini_pytorch/compiler/compiled.py
```

Explain:

```text
eager logical op
→ Graph Node
→ runtime signature
→ guards
→ cache hit or specialization miss
```

Historical branches: `v0.11`, `v0.12`.

---

## Round 11 — AOT backward and lowering

Run:

```bash
python examples/16_aot_autograd.py
python examples/17_lowering_fusion.py
```

Read:

```text
mini_pytorch/compiler/aot.py
mini_pytorch/compiler/analysis.py
mini_pytorch/compiler/lowering.py
```

Explain:

```text
forward Graph
→ saved-value ABI
→ backward Graph
→ analysis/transforms
→ LoweredProgram
```

Historical branches: `v0.13`, `v0.14`.

---

## Round 12 — Runtime planning and integrated architecture

Run:

```bash
python examples/18_runtime_performance.py
python examples/19_integrated_architecture.py
```

Read:

```text
mini_pytorch/compiler/benchmark.py
mini_pytorch/compiler/profiler.py
mini_pytorch/compiler/runtime_analysis.py
mini_pytorch/compiler/execution_plan.py
docs/ARCHITECTURE_V1.md
docs/PYTORCH_MAPPING.md
```

Explain:

```text
LoweredProgram = what executes
ExecutionPlan = how runtime resources are assigned
```

Then explain the complete path:

```text
Tensor expression
→ Storage / Device
→ Dispatcher / Backend
→ eager Autograd
→ Module / Optimizer / Data / Runtime / DDP
→ capture / guards / cache
→ AOT backward
→ lowering / fusion
→ MemoryPlan / ExecutionPlan
→ PlannedExecutor
```

Historical branch: `v0.15`; integrated release: `v1.0`.

---

# Final self-test

Without reading source, answer these questions:

1. Why can two Tensors share Storage but have different shape/stride?
2. Why is Autograd a Dispatcher policy rather than a CPU kernel property?
3. Why does `Module` not own differentiation?
4. Why does `Sampler` own shuffle order instead of `Dataset`?
5. What extra state must be restored for deterministic training resume?
6. Why does DDP synchronization stay outside optimizer algorithms?
7. Why is a guard failure not necessarily a compiler error?
8. What must forward retain for an explicit AOT backward Graph?
9. Why can fusion reduce instructions without guaranteeing lower latency?
10. Why does safe buffer reuse require last-use information?
11. What is the difference between `Graph`, `LoweredProgram`, and `ExecutionPlan`?
12. Which parts of mini-pytorch are analogies to PyTorch concepts rather than source-equivalent implementations?

If you can answer all twelve and point to the owning source module, the v1.0 learning goal is complete.
