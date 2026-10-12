# mini-pytorch v0.9 — Training Runtime / Serialization / Reproducibility

> 把“模型能训练”推进到“训练进程可以被暂停、保存、重新创建并确定性继续”。

**Package version:** `0.9.0`

**Validated release-candidate baseline:** `140/140 tests passed`

## 1. 本版本解决什么问题

只保存模型权重，并不能完整恢复训练。

真实训练过程还依赖：

```text
model parameters / buffers
optimizer momentum / moments
scheduler progress / current LR
random stream position
training counters
checkpoint schema metadata
```

v0.9 的核心目标是：

```text
training process
      ↓
collect complete logical state
      ↓
inspectable checkpoint
      ↓
construct fresh objects
      ↓
restore
      ↓
continue with the same future behavior
```

## 2. 相比 v0.8 的核心变化

v0.8：

```text
Tensor.backward()
      ↓
AutogradEngine
      ↓
Optimizer.step()
```

v0.9：

```text
Module
Optimizer
Scheduler
RNG
TrainingState
      │
      ▼
CheckpointState
      │
      ▼
metadata.json + tensors.npz
      │
      ▼
fresh model / optimizer / scheduler
      │
      ▼
restore
      │
      ▼
deterministic continuation
```

训练第一次被当成一个**完整状态机**来处理。

---

## 3. Framework RNG

新增：

```python
from mini_pytorch import (
    manual_seed,
    get_rng_state,
    set_rng_state,
)
```

关键区别：

```text
seed
→ 随机流从哪里开始

RNG state
→ 随机流现在走到哪里
```

因此 checkpoint 必须保存 state，而不是只记住 seed。

当前 framework-owned RNG 覆盖：

```text
Linear initialization
Conv2d initialization
Dropout(seed=None)
RandomSampler(seed=None)
```

显式 `seed=` 的 Dropout / RandomSampler 继续拥有组件私有随机流，用于保持旧版本语义。

---

## 4. TrainingState

新增：

```python
from mini_pytorch import TrainingState

state = TrainingState(
    epoch=2,
    global_step=731,
    batch_in_epoch=19,
    best_metric=0.91,
)
```

职责非常明确：

```text
TrainingState
→ 训练走到哪里
```

而不是：

```text
TrainingState
→ 保存所有 Tensor
```

model / optimizer / scheduler 各自继续使用自己的 `state_dict()`。

---

## 5. CheckpointState

逻辑 checkpoint：

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

它与文件格式分离。

因此：

```python
state = runtime.collect_checkpoint_state(...)
```

只是收集状态，不做磁盘 I/O。

---

## 6. 可检查的 checkpoint 格式

保存后：

```text
checkpoint/
├── metadata.json
└── tensors.npz
```

### metadata.json

可以直接打开阅读：

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

### tensors.npz

保存：

```text
model arrays
optimizer moments
buffer arrays
其他 ndarray state
```

复杂 dict / tuple / list 结构放在 JSON 中，并用显式 array reference 指向 NPZ。

没有使用 pickle 作为主要机制。

---

## 7. 保存 checkpoint

```python
from mini_pytorch import TrainingState, save_checkpoint

state = TrainingState(epoch=3, global_step=120)

save_checkpoint(
    "checkpoint",
    model,
    optimizer,
    scheduler,
    training_state=state,
    metadata={"experiment": "demo"},
)
```

逻辑路径：

```text
Module.state_dict
Optimizer.state_dict
Scheduler.state_dict
get_rng_state
TrainingState
       ↓
CheckpointState
       ↓
serialize
```

---

## 8. 恢复 checkpoint

重点不是“把旧对象反序列化回来”，而是：

```text
重新创建对象
      ↓
把 checkpoint state 应用到新对象
```

例如：

```python
model = build_model()
optimizer = optim.Adam(model.parameters(), lr=0.02)
scheduler = optim.StepLR(optimizer, step_size=2)

state = load_checkpoint(
    "checkpoint",
    model,
    optimizer,
    scheduler,
)
```

Restore 顺序：

```text
validate schema
      ↓
model
      ↓
optimizer
      ↓
scheduler
      ↓
TrainingState
      ↓
RNG state  ← 最后
```

RNG 最后恢复非常关键，因为 fresh model construction 本身可能消耗随机数。

---

## 9. Device-neutral checkpoint values

checkpoint 的 ndarray 是 host values，但不会强制目标模型回 CPU。

例如：

```python
model = build_model().to("edu")
load_checkpoint(..., model, optimizer)
```

状态加载使用现有 Parameter / Buffer Storage，因此：

```text
serialized NumPy value
      ↓
write into target Tensor Storage
      ↓
EDU device preserved
```

v0.9 已有专项测试验证 CPU checkpoint → EDU target restore。

---

## 10. 为什么 optimizer state 必须保存

以 Adam 为例：

```text
parameter
+ gradient
+ step
+ exp_avg
+ exp_avg_sq
+ LR/options
      ↓
next update
```

只有 parameter 一样，下一步训练也不一定一样。

因此：

```text
model weights only
≠
training state
```

---

## 11. 为什么 scheduler state 也必须保存

StepLR 等 scheduler 有自己的时间位置：

```text
last_epoch
_step_count
_last_lr
base_lrs
```

checkpoint 中如果存在 scheduler state，strict restore 要求提供目标 scheduler；不会静默忽略。

---

## 12. Deterministic Resume — 本版本核心验证

测试模型：

```text
Linear
  ↓
ReLU
  ↓
Dropout(seed=None)
  ↓
Linear
```

优化过程：

```text
Adam
+
StepLR
+
framework RNG
```

### Run A

```text
seed S
 ↓
train 1 ... N
 ↓
final state A
```

### Run B

```text
seed S
 ↓
train 1 ... K
 ↓
save
 ↓
construct fresh objects
 ↓
load
 ↓
train K+1 ... N
 ↓
final state B
```

CI 断言：

```text
losses A == losses B
model state A == B
optimizer state A == B
scheduler state A == B
TrainingState A == B
future RNG draws A == B
```

这才是 v0.9 对“恢复训练”的定义。

---

## 13. Resume boundary

v0.9 明确支持受控 step / epoch boundary 的确定性恢复。

虽然 `TrainingState` 有：

```text
batch_in_epoch
```

但本版本不声称可以恢复：

```text
任意已经消费一半的 multi-worker DataLoader iterator
```

如果 `RandomSampler(seed=None)` 在完整 epoch 边界继续，它的随机顺序由 framework RNG 控制并可随 checkpoint 恢复。

---

## 14. 写入安全

保存过程不是直接写最终目录：

```text
staging directory
      ↓
write metadata.json
write tensors.npz
fsync files
      ↓
publish by rename
```

覆盖已有 checkpoint 时会先移动旧目录到 backup；发布失败时尝试恢复旧版本。

这是教学型 atomic-ish persistence，而不是分布式生产 checkpoint 服务。

---

## 15. 严格失败语义

v0.9 会明确拒绝：

```text
unknown schema
missing metadata.json
missing tensors.npz
corrupt NPZ
wrong model parameter shape
optimizer group mismatch
scheduler state mismatch
existing path when overwrite=False
```

不会为了“加载成功”而偷偷 reshape 参数或丢弃重要 state。

---

## 16. 新增核心文件

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

重点修改：

```text
mini_pytorch/nn/linear.py
mini_pytorch/nn/conv.py
mini_pytorch/nn/dropout.py
mini_pytorch/data/sampler.py
mini_pytorch/__init__.py
pyproject.toml
```

---

## 17. 推荐阅读顺序

```text
runtime/rng.py
    ↓
runtime/state.py
    ↓
runtime/checkpoint.py
    ↓
nn/linear.py + conv.py + dropout.py
    ↓
data/sampler.py
    ↓
tests/test_runtime_checkpoint.py
    ↓
docs/TRAINING_RUNTIME.md
    ↓
examples/12_checkpoint_resume.py
```

---

## 18. 快速运行

```bash
pip install -e .
python -m unittest discover -s tests -v
python examples/12_checkpoint_resume.py
```

当前 release-candidate：

```text
140 tests
OK
```

相比 v0.8：

```text
123
  +17 runtime/checkpoint tests
=140
```

---

## 19. Explicit non-goals

v0.9 不实现：

- multi-worker arbitrary mid-epoch resume；
- distributed/sharded checkpoint；
- async checkpoint writer；
- remote object store；
- DDP/NCCL；
- compiler graph serialization；
- cryptographic checkpoint signing。

---

## 20. 下一版本

`v0.10` 将进入 **Distributed / DDP Concepts**：

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

下一步不再是“如何保存单机训练”，而是：

> 多个模型副本如何在每一步之后保持一致？
