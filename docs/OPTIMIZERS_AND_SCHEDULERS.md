# Optimizers and Learning-Rate Schedulers

The optimizer layer is built around one PyTorch-like boundary:

```text
Parameter Groups
      │
      ├── lr
      ├── momentum / betas / alpha / ...
      └── params
            │
            ▼
        Optimizer.step()
            │
            ▼
     parameter update

LRScheduler ───────► mutates param_group["lr"]
```

Schedulers only change the `lr` stored in parameter groups. They do not need
to know the optimizer's update equation, which keeps both systems extensible.

## Optimizers

- `SGD`: weight decay, momentum, dampening, Nesterov.
- `Adam`: first/second moments with bias correction.
- `AdamW`: Adam with decoupled weight decay.
- `Adamax`: infinity-norm Adam variant.
- `RMSprop`: exponential squared-gradient average, optional momentum/centering.
- `Adagrad`: cumulative squared-gradient adaptation.

All optimizers share `param_groups`, per-parameter `state`, `zero_grad()`,
`state_dict()`, `load_state_dict()`, and `learning_rates`.

### Adam vs AdamW

Adam with classical L2-style decay modifies the gradient before moment updates:

```text
g <- g + weight_decay * parameter
```

AdamW decouples that operation:

```text
parameter <- parameter * (1 - lr * weight_decay)
Adam update using the raw gradient
```

That distinction is one of the main reasons AdamW deserves its own optimizer.

## Learning-rate schedulers

Available schedulers:

```text
LRScheduler
├── LambdaLR
├── StepLR
├── MultiStepLR
├── ExponentialLR
├── CosineAnnealingLR
└── LinearLR

metric-driven:
└── ReduceLROnPlateau
```

Typical training loop:

```python
optimizer = optim.AdamW(model.parameters(), lr=3e-3)
scheduler = optim.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

for epoch in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

`LinearLR` can be used as a warm-up primitive:

```python
scheduler = optim.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=5,
)
```

`ReduceLROnPlateau` is different because it consumes a metric:

```python
scheduler = optim.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
)

scheduler.step(validation_loss)
```

## Checkpointing

```python
optimizer_state = optimizer.state_dict()
scheduler_state = scheduler.state_dict()

optimizer.load_state_dict(optimizer_state)
scheduler.load_state_dict(scheduler_state)
```

Optimizer checkpoints use positional parameter indices instead of Python
object ids, because object ids are not stable across processes.

## Extending the system

A new optimizer subclasses `Optimizer`, defines defaults, and implements
`step()`. A new epoch-based scheduler subclasses `LRScheduler` and implements
`get_lr()`. Neither extension requires modifying `Tensor`, `Module`, or the
training loop.
