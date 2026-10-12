# mini-pytorch v0.2

## 1. Version identity

- **Branch:** `v0.2`
- **Historical base commit:** `ed30650a2e3c9d913244e85ddd508000f62ee363`
- **Theme:** optimizer architecture and learning-rate scheduling
- **Previous version:** `v0.1`
- **Status:** frozen historical learning snapshot

v0.2 preserves the Tensor / Autograd / Module core from v0.1 and deepens the **optimization layer** so that training state, optimizer families and learning-rate evolution become first-class concepts.

---

## 2. Core difference from v0.1

### v0.1

```text
Tensor
  ↓
Autograd
  ↓
Module
  ↓
SGD / Adam
  ↓
training loop
```

### v0.2

```text
Tensor / Autograd / Module
          ↓
       Optimizer
      /    |     \
 parameter state  parameter groups
      \    |     /
       optimizer family
          ↓
      LR Scheduler
          ↓
learning-rate trajectory across epochs
```

The most important conceptual change is:

> The optimizer is no longer just `parameter -= lr * grad`; it is a stateful training subsystem whose state can evolve independently from the model.

---

## 3. Added in v0.2

### Richer optimizer family

Compared with v0.1, v0.2 adds or expands:

- enhanced SGD
  - momentum
  - dampening
  - Nesterov
  - weight decay
- Adam
- AdamW
- Adamax
- RMSprop
  - momentum
  - centered variant
- Adagrad

The implementation continues to share one extensible `Optimizer` abstraction rather than duplicating common state-management logic in every algorithm.

### Parameter groups

Optimizer parameter groups become a real architectural concept:

```python
optimizer = SGD([
    {"params": group_a, "lr": 1e-2},
    {"params": group_b, "lr": 1e-3},
])
```

This models the idea that different model parameters may use different optimization hyperparameters without changing the model itself.

### Optimizer checkpoint state

v0.2 strengthens optimizer serialization:

- `state_dict()`
- `load_state_dict()`
- per-parameter state restoration
- parameter-group configuration restoration

This is important because an optimizer such as Adam is not determined only by model weights. Its momentum estimates are part of the training state.

### Learning-rate scheduler subsystem

v0.2 introduces an independent scheduler architecture:

- `LRScheduler`
- `LambdaLR`
- `StepLR`
- `MultiStepLR`
- `ExponentialLR`
- `CosineAnnealingLR`
- `LinearLR`
- `ReduceLROnPlateau`

Schedulers modify optimizer parameter-group learning rates while leaving Tensor, Autograd and Module untouched.

Target dependency boundary:

```text
Model Parameters
      ↓
Optimizer
      ↑
LRScheduler
```

The scheduler depends on the optimizer; the optimizer does not need to know which scheduling policy is being used.

---

## 4. What remains unchanged from v0.1

v0.2 deliberately does **not** rewrite the framework core.

The following mental models remain stable:

- Tensor owns numerical data and autograd metadata
- differentiable operators build a dynamic graph
- `backward()` performs reverse-mode differentiation
- `Parameter` is a trainable Tensor registered by `Module`
- `Module` recursively exposes parameters
- training still follows forward → loss → backward → optimizer step

This stability is intentional. Each version should add one major learning boundary instead of changing every subsystem at once.

---

## 5. Representative training loop

v0.2 introduces the idea that optimizer and scheduler have independent lifecycles:

```python
optimizer = AdamW(
    model.parameters(),
    lr=3e-3,
    weight_decay=1e-2,
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-5,
)

for epoch in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

The important lesson is that these two calls do different things:

```text
optimizer.step()  -> update parameters
scheduler.step()  -> update optimizer learning rates
```

---

## 6. Important source files

Focus on these files after understanding v0.1:

```text
mini_pytorch/optim/optimizer.py
        ↓
mini_pytorch/optim/sgd.py
mini_pytorch/optim/adam.py
mini_pytorch/optim/adamw.py
mini_pytorch/optim/rmsprop.py
mini_pytorch/optim/adagrad.py
mini_pytorch/optim/adamax.py
        ↓
mini_pytorch/optim/lr_scheduler.py
```

Also read:

```text
docs/OPTIMIZERS_AND_SCHEDULERS.md
examples/05_lr_schedulers.py
```

---

## 7. Learning goals

By the end of v0.2, the reader should be able to explain:

1. Why SGD momentum needs optimizer state.
2. Why Adam has first- and second-moment buffers.
3. Why AdamW decouples weight decay from the gradient update.
4. Why optimizer state must be checkpointed together with model state.
5. Why parameter groups exist.
6. Why schedulers modify optimizer learning rates instead of model parameters directly.
7. Why `ReduceLROnPlateau` is metric-driven while `StepLR` is epoch-driven.

---

## 8. What v0.2 still does not implement

Compared with real PyTorch, the following remain outside this version:

- Dataset abstraction as a complete subsystem
- Sampler / BatchSampler
- MNIST end-to-end pipeline
- DataLoader iterator lifecycle
- multiprocessing workers
- CrossEntropy classification path as the main training benchmark
- explicit Storage / stride / storage offset
- contiguous / non-contiguous Tensor semantics
- Conv2d / pooling
- device/backend abstraction

These are postponed because they belong to different conceptual layers.

---

## 9. Version delta summary

```text
v0.1
├── Tensor / Autograd
├── Module / Parameter
├── Linear / activations / MSE
├── SGD
└── Adam

          ↓ v0.2 adds

v0.2
├── everything above
├── richer SGD behavior
├── AdamW
├── Adamax
├── RMSprop
├── Adagrad
├── stronger optimizer state serialization
├── parameter-group usage
└── LR scheduler subsystem
```

So the short version is:

> **v0.1 teaches how training works. v0.2 teaches how optimization state evolves during training.**

---

## 10. Next version

`v0.3` shifts attention away from optimization and into the **data pipeline**:

```text
Dataset
  ↓
Sampler
  ↓
BatchSampler
  ↓
DataLoader
  ↓
MNIST
  ↓
classification training
```

For the exact delta, read `version.md` on the `v0.3` branch.
