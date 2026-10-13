# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行的 Python / NumPy 代码，按真实 PyTorch 的架构依赖顺序，一层一层拆开核心机制。

**Latest integrated version:** `v0.10.0`

**Current main milestone:** Distributed / DDP Concepts

**Next milestone:** `v0.11` Graph IR / Eager-to-Graph Capture

mini-pytorch 不追求复制完整 PyTorch API，而是持续回答：

```text
Tensor 是什么？
      ↓
Autograd 为什么能自动求导？
      ↓
Optimizer / Scheduler 如何组织训练动力学？
      ↓
Dataset / Sampler / DataLoader 为什么分层？
      ↓
Storage / stride / view / version counter 为什么重要？
      ↓
operator 为什么需要 Dispatcher / redispatch？
      ↓
复杂 NN operator 如何建立在 Dispatcher 上？
      ↓
Device / Backend 如何与 Tensor / Storage 分离？
      ↓
AutogradEngine 如何管理 graph lifetime / hooks？
      ↓
完整训练状态如何 checkpoint 并确定性恢复？
      ↓
多个 replica 如何通过 DDP 同步 gradient？
      ↓
eager execution 如何变成可检查、可重放的 Graph IR？
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
| `v0.10` | Distributed / DDP Concepts | 多个 replica 如何同步 gradient 并保持一致？ |
| `v0.11` | Graph IR / Eager-to-Graph Capture | eager operator sequence 如何变成显式 Graph？ |

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
                      persistent requested grad

Dataset → Sampler → BatchSampler → DataLoader
           │
           └──────────────→ DistributedSampler

Module / Parameter → Optimizer → LR Scheduler
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

Replicated Modules
        │
        ▼
leaf Parameter hooks
        │
        ▼
      Reducer
        │
  GradientBucket(s)
        │
        ▼
    ProcessGroup
        │
     all_reduce
        │
        ▼
synchronized gradients
```

`EDU` 是确定性的教学模拟 backend：拥有独立 Device / Backend / DispatchKey，但仍以 NumPy host arrays 为数值载体。它用于验证架构边界，不代表 CUDA / GPU 加速。

v0.10 的 distributed world 同样是**教学型单进程 transport simulation**，用于理解 rank / world-size / collective ordering / buckets / Reducer，而不是伪装成 NCCL 或真实多机。

## 3. v0.10 最新能力

### Rank / world size

```python
from mini_pytorch.distributed import create_process_group

g0, g1 = create_process_group(2)
```

每个 rank-local `ProcessGroup` 拥有：

```text
rank
world_size
next collective sequence
```

但共享同一个 in-process world。

### Async-style Work handle

```python
w0 = g0.all_reduce([1.0, 3.0], op="mean")
```

只有 rank 0 提交时，Work 未完成；不会阻塞等待尚未执行的 rank 1。

当所有 rank 对同一 collective sequence 提交后，Work 才能读取结果。

同一 sequence 必须匹配：

```text
operation
shape
dtype
```

### DistributedSampler

```python
DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=0,
)
```

核心：

```text
same global ordering
      ↓
strided rank partition
```

支持 padding / drop_last / `set_epoch(epoch)`。

### DistributedDataParallel

```text
local backward
      ↓
Parameter hook
      ↓
Reducer.mark_variable_ready
      ↓
GradientBucket
```

DDP 不修改 `Function.backward()`，也不让 backend kernel 知道 distributed runtime。

### Explicit synchronization barrier

由于 rank 在一个 Python 进程里顺序执行，v0.10 不会在 backward hook 里阻塞等待 peers。

采用：

```text
Phase 1
all replica backward
→ stage gradients

Phase 2
synchronize_ddp([...])
→ launch all rank collectives
→ complete Work handles
→ unflatten averaged gradients
```

### Gradient buckets

Reducer 按反向 parameter registration order 构建 deterministic buckets，并记录：

```text
name
shape
numel
offset
dtype
device
```

不同 dtype/device 不进入同一 bucket。

### Global-batch reference proof

最重要的 CI 验证不是“两个 replica 看起来相等”，而是：

```text
mean(rank-local gradients)
==
gradient(concatenated global batch)
```

在 equal local batch size + mean loss 条件下，两个 DDP replica 与一个 global-batch reference model 执行相同 SGD step 后参数一致。

### EDU placement

EDU 模型在 DDP sync 后仍保持 EDU Parameter / Storage identity；collective 只是 NumPy host-value simulation。

## 4. 当前测试基线

```text
v0.9   140 tests
v0.10  159 tests
```

v0.10 新增 19 个 tests，覆盖：

- rank / world-size；
- sum / mean all-reduce；
- incomplete Work；
- collective op / shape / dtype mismatch；
- DistributedSampler；
- bucket construction；
- gradient-ready hooks；
- explicit barrier；
- missing rank；
- unused parameter；
- stale grad iteration boundary；
- initial state broadcast；
- EDU placement；
- global-batch reference equivalence。

完整回归：

```bash
pip install -e .
python -m unittest discover -s tests -v
```

## 5. 推荐源码学习顺序

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
v0.10 distributed/process_group.py
      → distributed/sampler.py
      → distributed/reducer.py
      → distributed/ddp.py
      → docs/DISTRIBUTED_DDP.md
```

对应后期教学示例：

```bash
python examples/09_cnn_mnist.py
python examples/10_device_backend.py
python examples/11_autograd_engine.py
python examples/12_checkpoint_resume.py
python examples/13_ddp_simulation.py
```

## 6. 当前明确非目标

截至 v0.10，项目刻意还没有实现：

- real CUDA / NCCL / Gloo / MPI；
- true multi-process / multi-node runtime；
- true higher-order Autograd；
- DDP communication/backward overlap；
- `no_sync()` / unused-parameter graph detection；
- FSDP / ZeRO；
- eager-to-graph capture；
- graph guards / compilation cache；
- AOTAutograd；
- compiler lowering / fusion；
- production distributed checkpoint service。

这些会按 `Plan.md` 的架构依赖顺序逐步进入。

## 7. 下一版：v0.11 Graph IR / Eager-to-Graph Capture

下一阶段核心问题：

> 一次 eager Tensor execution，如何被表示为显式的 `Graph / Node / Value`，随后再次解释执行，并与原 eager 结果保持一致？

目标路径：

```text
Tensor eager calls
      ↓
Dispatcher
      ↓
capture observer
      ↓
Value / Node / Graph IR
      ↓
GraphInterpreter
      ↓
Dispatcher replay
      ↓
eager-vs-graph equivalence
```

v0.11 会优先实现可读的 IR 和 replay，而不是直接做优化器、fusion 或“编译器魔法”。

详细设计与验收任务见 `Plan.md`。
