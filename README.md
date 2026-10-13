# mini-pytorch v0.10 — Distributed / DDP Concepts

> 用一个确定性的单进程模拟器，把 PyTorch DDP 最核心的架构链真正跑通：**数据分片 → local backward → gradient-ready hooks → buckets → ProcessGroup all-reduce → replica-consistent gradients**。

**Package version:** `0.10.0`

**Current validated baseline:** `159/159 tests passed`

## 1. 本版本核心问题

两个完全相同的模型副本，如果各自看到不同数据：

```text
rank 0                     rank 1
  │                          │
batch A                    batch B
  │                          │
backward                   backward
  │                          │
grad_A                     grad_B
```

直接执行 optimizer：

```text
w0' = w - lr * grad_A
w1' = w - lr * grad_B
```

通常会导致：

```text
w0' != w1'
```

v0.10 构建的解决链路：

```text
local gradients
      ↓
Reducer buckets
      ↓
ProcessGroup all_reduce(mean)
      ↓
shared averaged gradients
      ↓
optimizer.step()
      ↓
replicas remain equal
```

## 2. 相比 v0.9 的变化

v0.9 解决：

```text
一个训练进程如何 checkpoint / restore？
```

v0.10 解决：

```text
多个 replica 如何同步梯度？
```

完整结构：

```text
                     Dataset
                        │
                        ▼
              DistributedSampler
             ┌──────────┴──────────┐
             ▼                     ▼
          rank 0                  rank 1
             │                     │
             ▼                     ▼
         Module 0               Module 1
             │                     │
             ▼                     ▼
      AutogradEngine 0      AutogradEngine 1
             │                     │
             ▼                     ▼
      Parameter hooks       Parameter hooks
             │                     │
             ▼                     ▼
         Reducer 0              Reducer 1
             │                     │
             ▼                     ▼
      GradientBucket(s)     GradientBucket(s)
             │                     │
             └──────────┬──────────┘
                        ▼
                  ProcessGroup
                        │
                     all_reduce
                        │
                        ▼
               averaged gradients
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        optimizer 0            optimizer 1
```

## 3. 新增模块

```text
mini_pytorch/distributed/
├── __init__.py
├── process_group.py
├── sampler.py
├── reducer.py
└── ddp.py
```

新增：

```text
tests/test_distributed.py
docs/DISTRIBUTED_DDP.md
examples/13_ddp_simulation.py
```

## 4. ProcessGroup

创建一个两 rank 的模拟 world：

```python
from mini_pytorch.distributed import create_process_group

g0, g1 = create_process_group(2)
```

每个 rank 拥有独立：

```text
rank
world_size
next collective sequence
```

但共享同一个 in-process collective world。

### Work handle

```python
w0 = g0.all_reduce([1.0, 3.0], op="mean")
```

此时只有 rank 0 提交：

```text
w0.is_completed == False
```

不会阻塞等待 rank 1。

当：

```python
w1 = g1.all_reduce([3.0, 5.0], op="mean")
```

两边都可以取得：

```text
[2.0, 4.0]
```

## 5. Collective order 是协议

同一个 sequence 上，所有 rank 必须一致：

```text
operation
shape
dtype
```

例如：

```text
rank0 sequence 2: mean, shape=(128,), float64
rank1 sequence 2: sum,  shape=(128,), float64
```

会明确失败。

这说明 distributed collective 不是几个互不相关的本地函数调用，而是一个多参与者协议。

## 6. DistributedSampler

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=4,
    rank=2,
    shuffle=True,
    seed=123,
)
```

职责只有：

```text
决定这个 rank 看哪些 index
```

不加载数据。

核心逻辑：

```text
create global ordering
      ↓
indices[rank:total_size:num_replicas]
```

支持：

- deterministic shuffle；
- `set_epoch(epoch)`；
- padding；
- `drop_last`。

## 7. DDP 不修改 Autograd 公式

`DistributedDataParallel` 在每个 trainable Parameter 上注册 hook：

```text
leaf gradient ready
      ↓
Tensor hook
      ↓
Reducer.mark_variable_ready(...)
```

hook 只记录 local gradient ready。

不会在 `Function.backward()` 中加入 distributed 特判，也不会让 backend kernel 知道 DDP。

## 8. 为什么不能在 hook 内等待其他 rank

本项目当前模拟：

```text
执行 rank0 Python
然后执行 rank1 Python
```

如果 rank0 backward hook：

```text
等待 rank1
```

程序就永远执行不到 rank1。

因此采用明确的两阶段模型：

```text
Phase 1
rank0 backward → stage gradients
rank1 backward → stage gradients

Phase 2
synchronize_ddp([ddp0, ddp1])
      ↓
all ranks launch collectives
      ↓
all works complete
      ↓
unflatten averaged grads
```

## 9. Gradient buckets

```python
ddp = DistributedDataParallel(
    model,
    group,
    bucket_cap_numel=1024,
)
```

Reducer 按**反向 parameter registration order**构建 bucket。

原因：reverse-mode backward 中，后面的 layer 往往更早产生梯度。

每个 bucket 记录：

```text
parameter name
shape
numel
offset
dtype
device
```

不同 dtype/device 不会混在同一 bucket。

## 10. Flatten / Reduce / Unflatten

```text
weight.grad  ┐
bias.grad    ├─ flatten → one buffer
other.grad   ┘
                 ↓
             all_reduce
                 ↓
             flat result
                 ↓
            offsets/shapes
                 ↓
        write Parameter.grad
```

optimizer 完全不知道这个过程发生过。

它只看到同步后的 `.grad`。

## 11. 最重要的 correctness proof

v0.10 CI 真正验证：

```text
rank0 local batch size = 2
rank1 local batch size = 2
local MSE reduction = mean
      ↓
average(local gradients)
```

等于：

```text
single reference model
concatenate 4 samples
MSE reduction = mean
      ↓
reference gradient
```

然后三个模型执行相同 SGD update 后：

```text
replica0 == replica1 == reference
```

不是只检查“两个 replica 彼此相同”，还检查它们是否等价于正确的 global-batch 数学结果。

## 12. Initial state synchronization

提供：

```python
broadcast_module_state(
    [ddp0, ddp1],
    src=0,
)
```

用于让所有 replica 的 Parameters/Buffers 从同一个源开始。

## 13. Iteration contract

v0.10 推荐训练周期：

```text
optimizer.zero_grad()
      ↓
DDP forward
      ↓
loss.backward()
      ↓
synchronize_ddp(all_replicas)
      ↓
optimizer.step()
```

如果旧 `.grad` 没清除就进入下一次 DDP forward，会明确报错。

这一版尚不实现 `no_sync()` 多次 backward accumulation。

## 14. Unused parameter

Reducer 期待所有 trainable Parameters 都 ready。

未参与 graph 的 Parameter 会导致同步失败，并列出 missing parameter name。

真实 DDP 的 unused-parameter detection 属于更复杂的 graph/runtime 逻辑，本版明确不实现。

## 15. EDU backend

专项测试验证：

```text
EDU Parameter
      ↓
local backward
      ↓
DDP sync
      ↓
EDU Parameter 仍保持 EDU
```

collective transport 仍是 NumPy host simulation，不代表 GPU communication。

## 16. 测试基线

```text
v0.9
140 tests
   │
   │ +19 distributed tests
   ▼
v0.10
159 tests
OK
```

新增覆盖：

- rank/world-size；
- mean/sum all-reduce；
- incomplete Work；
- operation/shape/dtype mismatch；
- DistributedSampler partition/padding/drop_last/set_epoch；
- deterministic reversed buckets；
- gradient-ready hooks；
- explicit sync；
- missing rank；
- unused parameter；
- stale gradient boundary；
- initial state broadcast；
- EDU placement；
- global-batch reference equivalence。

## 17. 推荐阅读顺序

```text
mini_pytorch/distributed/process_group.py
        ↓
mini_pytorch/distributed/sampler.py
        ↓
mini_pytorch/distributed/reducer.py
        ↓
mini_pytorch/distributed/ddp.py
        ↓
tests/test_distributed.py
        ↓
docs/DISTRIBUTED_DDP.md
        ↓
examples/13_ddp_simulation.py
```

## 18. 快速运行

```bash
pip install -e .
python -m unittest discover -s tests -v
python examples/13_ddp_simulation.py
```

## 19. Explicit non-goals

v0.10 不实现：

- multiprocessing / multi-node；
- NCCL / Gloo / MPI；
- network transport；
- CUDA collectives；
- communication/backward real overlap；
- `no_sync()`；
- unused-parameter graph detection；
- uneven-input join；
- FSDP / ZeRO；
- distributed checkpoint；
- production fault tolerance。

## 20. 下一版本

`v0.11` 将进入 **Graph IR / Eager-to-Graph Capture**。

核心问题从：

```text
多个 replica 如何同步？
```

转向：

```text
一次 eager Tensor execution
如何被表示成显式 Graph / Node / Value，
并再次解释执行？
```

目标路径：

```text
eager operators
      ↓
trace/capture
      ↓
Graph IR
      ↓
graph interpreter
      ↓
eager-vs-graph equivalence
```
