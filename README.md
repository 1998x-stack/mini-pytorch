# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行的 Python/NumPy 代码，按真实 PyTorch 的架构依赖顺序，一层一层拆开核心机制。

**Latest integrated version:** `v0.9.0`

**Current main milestone:** Training Runtime / Serialization / Reproducibility

**Next milestone:** `v0.10` Distributed / DDP Concepts

mini-pytorch 不追求复制完整 PyTorch API，而是持续回答：

```text
Tensor 是什么？
      ↓
Autograd 为什么能自动求导？
      ↓
Optimizer / Scheduler 如何保存训练动力学？
      ↓
Dataset / Sampler / DataLoader 为什么分层？
      ↓
Storage / stride / view / version counter 有什么意义？
      ↓
operator 为什么需要 Dispatcher / redispatch？
      ↓
复杂 NN operator 如何建立在 Dispatcher 上？
      ↓
Device / Backend 如何与 Tensor / Storage 分离？
      ↓
AutogradEngine 如何管理 graph lifetime / hooks / retained grad？
      ↓
完整训练状态如何 checkpoint 并确定性恢复？
      ↓
多个模型副本如何同步 gradient 并保持一致？
```

## 1. 版本学习地图

| Version | 核心主题 | 最重要的问题 |
|---|---|---|
| `v0.1` | Tensor + Dynamic Autograd | `backward()` 如何自动求导？ |
| `v0.2` | Optimizer + LR Scheduler | 优化状态为什么独立于模型？ |
| `v0.3` | Dataset + Sampler + DataLoader + MNIST | 数据为什么拆成采样、批处理与加载？ |
| `v0.4` | Storage + stride + views + version counter | Tensor 为什么不等于 ndarray？ |
| `v0.5` | Operator Schema + Dispatcher + Redispatch | 一个 operator 为什么需要路由层？ |
| `v0.6` | Conv2d / Pooling / Dropout / BatchNorm | 复杂 NN op 如何复用 Dispatcher？ |
| `v0.7` | Device / Backend | 多 backend 为什么不能靠散落的 `if device`？ |
| `v0.8` | Advanced Autograd Engine | graph lifetime / hooks / retain_graph 如何工作？ |
| `v0.9` | Training Runtime / Checkpoint / RNG | 怎样暂停训练并确定性继续？ |
| `v0.10` | Distributed / DDP Concepts | 多个 replica 如何同步 gradient？ |

每个历史 version branch 都保留：

```text
README.md   → 本版本学习入口
version.md  → 与上一版本的精确 delta
Agents.md   → 本版本工程边界与冻结规则
```

`main` 始终代表最新完成并验证的框架状态。

## 2. 当前完整架构

```text
                           Tensor
                shape / stride / dtype / device
                             │
                             ▼
                           Storage
              backend-owned allocation + version
                             │
                             ▼
                       logical operator
                             │
                             ▼
                         Dispatcher
            policy keys + backend dispatch keys
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           Autograd        NoGrad        CPU / EDU
              │                              │
              └──────── redispatch ──────────┘
                             │
                             ▼
                      numerical kernel
                             │
                             ▼
                       Tensor / GradFn
                             │
                             ▼
                       AutogradEngine
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        local grad flow  graph lifetime    hooks
                             │
                             ▼
                   requested persistent grad

Dataset → Sampler → BatchSampler → DataLoader

Module / Parameter
        ↓
Optimizer
        ↓
LR Scheduler
        │
        ├───────────────┐
        ▼               ▼
 TrainingState      framework RNG
        │               │
        └───────┬───────┘
                ▼
         CheckpointState
                ▼
     metadata.json + tensors.npz
                ▼
       validated restoration
```

`EDU` 是确定性的教学模拟 backend：拥有独立 Device / Backend / DispatchKey，但仍以 NumPy host arrays 为数值载体。它用于验证架构边界，不代表 CUDA/GPU 加速。

## 3. v0.9 最新能力

### Framework RNG

```python
from mini_pytorch import manual_seed, get_rng_state, set_rng_state
```

v0.9 明确区分：

```text
seed       → 随机流从哪里开始
RNG state  → 随机流当前走到哪里
```

framework-owned RNG 已用于：

```text
Linear initialization
Conv2d initialization
Dropout(seed=None)
RandomSampler(seed=None)
```

### TrainingState

```python
from mini_pytorch import TrainingState

state = TrainingState(
    epoch=2,
    global_step=731,
    batch_in_epoch=19,
)
```

它描述训练运行位置，不与模型 Tensor 状态混在一起。

### CheckpointState

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

状态收集、文件序列化和对象恢复是三个独立阶段。

### Inspectable checkpoint

```text
checkpoint/
├── metadata.json
└── tensors.npz
```

JSON 保留可读结构和 schema；NumPy arrays 放在 NPZ 中。没有把整个训练状态藏进一个 pickle blob。

### Restore ordering

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

RNG 最后恢复，确保重新创建模型时消耗的随机数不会改变恢复后的未来随机序列。

### Device-neutral checkpoint values

checkpoint 保存的是 NumPy values；恢复时写入目标对象已有 Storage。因此 CPU 保存的 checkpoint 可以恢复到已经 `.to("edu")` 的模型，而不会偷偷把 Parameter 移回 CPU。

### Deterministic resume proof

v0.9 的核心集成测试比较：

```text
Run A
seed → train 1 ... N

Run B
seed → train 1 ... K
     → save
     → fresh model / optimizer / scheduler
     → load
     → train K+1 ... N
```

并验证完全一致的：

```text
loss trajectory
model state
Adam moments/counters
StepLR state/current LR
TrainingState
future framework RNG draws
```

当前验证基线：

```text
140 tests
OK
```

## 4. 推荐源码学习顺序

```text
v0.1  autograd.py → tensor.py → ops.py
v0.2  optim/optimizer.py → optim/* → lr_scheduler.py
v0.3  data/dataset.py → sampler.py → dataloader.py → mnist.py
v0.4  storage.py → tensor.py → docs/TENSOR_MEMORY.md
v0.5  dispatcher.py → operator_registry.py → docs/DISPATCHER.md
v0.6  kernels.py → nn_ops.py → nn/* → docs/CNN_OPERATORS.md
v0.7  device.py → backend.py → storage.py → dispatcher.py → docs/DEVICE_BACKEND.md
v0.8  autograd.py → autograd_engine.py → tensor.py → docs/AUTOGRAD_ENGINE.md
v0.9  runtime/rng.py → runtime/state.py → runtime/checkpoint.py → docs/TRAINING_RUNTIME.md
```

对应教学示例：

```bash
python examples/06_mnist_mlp.py
python examples/07_tensor_memory.py
python examples/08_dispatcher.py
python examples/09_cnn_mnist.py
python examples/10_device_backend.py
python examples/11_autograd_engine.py
python examples/12_checkpoint_resume.py
```

完整回归：

```bash
pip install -e .
python -m unittest discover -s tests -v
```

## 5. 当前明确非目标

截至 v0.9，项目刻意还没有实现：

- real CUDA / NCCL；
- asynchronous streams/events；
- true higher-order Autograd；
- multiprocessing distributed runtime；
- DDP gradient buckets；
- eager-to-graph capture；
- compiler lowering/fusion；
- distributed/sharded checkpoint service。

这些会按架构依赖顺序进入后续版本。

## 6. 下一版：v0.10 Distributed / DDP Concepts

下一阶段要回答：

> 如果有多个模型 replica，各自看不同数据并独立 backward，怎样同步 gradient，才能让所有 replica 在 optimizer step 后保持一致？

目标架构：

```text
local backward
     ↓
parameter gradient-ready hooks
     ↓
DDP Reducer
     ↓
gradient buckets
     ↓
ProcessGroup.all_reduce
     ↓
averaged gradients
     ↓
optimizer.step on each replica
     ↓
replica parameters remain equivalent
```

v0.10 会优先构建单进程、确定性的教学型 collective simulation，理解 `rank / world_size / DistributedSampler / ProcessGroup / AllReduce / Reducer / bucket` 的职责边界；不会直接跳到 NCCL 或真实多机网络。

详细设计与验收任务见 `Plan.md`。
