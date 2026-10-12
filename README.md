# mini-pytorch v0.2 — Optimizer State + Learning-Rate Scheduling

> 在 v0.1 的 Tensor / Autograd / Module 基础上，专门拆解 PyTorch 的优化器与学习率调度体系。

**版本主题：** stateful optimizers + parameter groups + checkpoint state + LR scheduler

**状态：** frozen historical learning snapshot

## 1. v0.2 相比 v0.1 新增了什么

v0.1 重点回答：

```text
梯度如何产生？
```

v0.2 进一步回答：

```text
梯度产生后，训练状态如何长期演化？
```

核心结构：

```text
Parameter
   │
   ├────────────► grad
   │
Optimizer
   ├────────────► parameter groups
   ├────────────► per-parameter state
   └────────────► state_dict
                         ▲
                         │
                    LRScheduler
                    modifies lr
```

关键认知：

> Optimizer 不是 `p -= lr * grad` 的一行代码，而是拥有独立状态生命周期的训练子系统。

## 2. Optimizer family

v0.2 包含：

- SGD
  - momentum
  - dampening
  - Nesterov
  - weight decay
- Adam
- AdamW
- Adamax
- RMSprop
  - momentum
  - centered
- Adagrad

所有算法共享统一的：

```text
Optimizer
├── param_groups
├── state
├── zero_grad()
├── state_dict()
└── load_state_dict()
```

## 3. Parameter Groups

```python
optimizer = optim.SGD(
    [
        {"params": model_a.parameters(), "lr": 1e-2},
        {"params": model_b.parameters(), "lr": 1e-3},
    ],
    lr=1e-2,
)
```

这解释了真实训练中的：

- different learning rates
- different weight decay
- fine-tuning
- layer-wise learning-rate decay
- frozen / unfrozen parameter groups

## 4. 为什么 Optimizer state 必须保存

以 Adam 为例：

```text
parameter
   │
   ├── exp_avg      ← first moment
   ├── exp_avg_sq   ← second moment
   └── step
```

仅保存 model weights 并不能恢复完整训练状态，因此 v0.2 强化了：

```text
model state
+
optimizer state
+
scheduler state
```

之间的边界。

## 5. LR Scheduler

v0.2 引入独立 Scheduler subsystem：

- `LRScheduler`
- `StepLR`
- `MultiStepLR`
- `ExponentialLR`
- `CosineAnnealingLR`
- `LinearLR`
- `LambdaLR`
- `ReduceLROnPlateau`

典型训练循环：

```python
optimizer = optim.AdamW(model.parameters(), lr=3e-3)
scheduler = optim.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

这里：

```text
optimizer.step() → 改参数
scheduler.step() → 改 optimizer.param_groups[*]["lr"]
```

## 6. 推荐源码阅读顺序

先掌握 v0.1，再阅读：

```text
mini_pytorch/optim/optimizer.py
        ↓
mini_pytorch/optim/sgd.py
        ↓
mini_pytorch/optim/adam.py
mini_pytorch/optim/adamw.py
mini_pytorch/optim/rmsprop.py
mini_pytorch/optim/adagrad.py
mini_pytorch/optim/adamax.py
        ↓
mini_pytorch/optim/lr_scheduler.py
        ↓
tests/test_optim_extended.py
        ↓
tests/test_lr_scheduler.py
```

推荐同时查看：

```text
docs/OPTIMIZERS_AND_SCHEDULERS.md
examples/05_lr_schedulers.py
```

## 7. 快速运行

```bash
pip install -e .
python -m unittest discover -s tests -v
python examples/05_lr_schedulers.py
```

## 8. 学完这一版应该能解释

1. Momentum 为什么需要 buffer；
2. Adam 为什么有一阶/二阶矩；
3. Adam 和 AdamW 的 weight decay 差异；
4. parameter group 为什么属于 Optimizer；
5. 为什么 scheduler 不直接修改 model；
6. 为什么 optimizer checkpoint 不能只存超参数；
7. `ReduceLROnPlateau` 为什么和普通 epoch scheduler 不一样。

## 9. 本版本不做什么

v0.2 不引入：

- Dataset / Sampler / DataLoader
- MNIST end-to-end classification
- explicit Storage / stride / view semantics
- inplace version counter
- Dispatcher
- Conv2d / device backend / DDP / compiler

## 10. 下一版本

`v0.3` 转向训练系统的另一侧：**Dataset → Sampler → BatchSampler → DataLoader → classification → MNIST**。

详细版本边界见 [`version.md`](./version.md)。