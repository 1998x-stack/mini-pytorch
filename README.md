# mini-pytorch — Architecture-Driven PyTorch Learning Project

> 用可运行的 Python/NumPy 代码，按真实 PyTorch 的架构依赖顺序，一层一层拆开核心机制。

**Latest integrated version:** `v0.8.0`

**Current main milestone:** Advanced Autograd Engine

**Next milestone:** `v0.9` Training Runtime / Serialization / Reproducibility

mini-pytorch 不追求复制完整 PyTorch API，而是持续回答这些问题：

```text
Tensor 是什么？
      ↓
Autograd 如何构图和 backward？
      ↓
Module / Optimizer 如何组织训练状态？
      ↓
Dataset / Sampler / DataLoader 为什么拆开？
      ↓
Storage / stride / view 为什么重要？
      ↓
operator 为什么需要 Dispatcher？
      ↓
复杂 NN operator 如何建立在 Dispatcher 上？
      ↓
Tensor 的 device/backend identity 如何建模？
      ↓
Autograd Engine 如何管理 graph lifetime / hooks / repeated backward？
      ↓
训练到底有哪些状态必须被 checkpoint 才能确定性恢复？
```

## 1. 版本学习地图

| Version | 核心主题 | 最重要的问题 |
|---|---|---|
| `v0.1` | Tensor + Dynamic Autograd | `backward()` 如何自动求导？ |
| `v0.2` | Optimizer + LR Scheduler | 优化状态为什么独立于模型？ |
| `v0.3` | Dataset + Sampler + DataLoader + MNIST | 数据为什么拆成 dataset / sampling / batching？ |
| `v0.4` | Storage + stride + views + version counter | Tensor 为什么不等于 ndarray？ |
| `v0.5` | Operator Schema + Dispatcher + Redispatch | 一个 `x + y` 为什么需要路由层？ |
| `v0.6` | Conv2d / Pooling / Dropout / BatchNorm | 复杂 NN operator 如何复用 Dispatcher？ |
| `v0.7` | Device / Backend | 多 backend 为什么不能靠散落的 `if device`？ |
| `v0.8` | Advanced Autograd Engine | graph lifetime / retain_graph / hooks 如何工作？ |
| `v0.9` | Training Runtime / Checkpoint / RNG | 怎样暂停训练并确定性继续？ |

每个历史版本 branch 都包含：

```text
README.md   → 本版本学习入口
version.md  → 与上一版的精确 delta
Agents.md   → 本版本工程约束
```

## 2. 当前 main 架构

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
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Autograd       NoGrad       CPU / EDU
             │                           │
             └────── redispatch ─────────┘
                           │
                           ▼
                    numerical kernel
                           │
                           ▼
                      Tensor / GradFn
                           │
                           ▼
                   Context / GradFn DAG
                           │
                           ▼
                     AutogradEngine
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
       local grad flow  graph lifetime  gradient hooks
             │          retain_graph      │
             ▼                            ▼
       leaf .grad                 retained non-leaf .grad

nn.Module → functional NN operator → Dispatcher

Dataset → Sampler → BatchSampler → DataLoader

Module / Parameter → Optimizer → LR Scheduler
```

`EDU` 是确定性的模拟教学 backend：拥有独立 Device / Backend / DispatchKey，但底层仍使用 NumPy host memory。它用于验证架构，不代表 CUDA/GPU 加速。

## 3. v0.8 最新能力

### 显式 Autograd Engine

```text
Tensor.backward()
      ↓
AutogradEngine.backward()
      ↓
Topology
      ↓
Engine-local gradient accumulation
      ↓
Hooks
      ↓
Function.backward VJP
      ↓
release / retain graph
```

Backward traversal 不再写死在 Tensor 里。

### Graph lifetime

默认：

```python
loss.backward()
```

成功后释放 backward-only Context state，因此再次执行同一 graph 会明确失败。

需要重复执行时：

```python
loss.backward(retain_graph=True)
loss.backward()
```

### Non-leaf `retain_grad()`

```python
hidden = x * x
hidden.retain_grad()
```

默认 non-leaf gradient 只是 Engine transient state；调用 `retain_grad()` 后才持久化到 `hidden.grad`。

### Gradient hooks

```python
handle = tensor.register_hook(fn)
handle.remove()
```

hook 可以观察 gradient，也可以返回 replacement gradient 改变上游梯度流。

### Custom Function

`Function / Context / GradFn / AutogradEngine` 已作为公开教学接口。

Custom Function 与内置 Function 共用：

```text
Function.apply → Context / GradFn → AutogradEngine
```

### Higher-order gradient boundary

```python
loss.backward(create_graph=True)
```

当前明确抛出 `NotImplementedError`，因为 backward formulas 仍产生 `np.ndarray`，尚未建立可微分的 backward Tensor graph。

## 4. 当前测试基线

```text
v0.8
123 tests
OK
```

其中 v0.8 新增 20 个 Autograd Engine tests，覆盖：

- graph release / repeated backward；
- `retain_graph`；
- shared-node accumulation；
- non-leaf `retain_grad()`；
- gradient hook order / transform / removal；
- custom Function；
- saved-version safety；
- CPU / EDU parity；
- explicit higher-order limitation。

## 5. 推荐学习顺序

```text
v0.1  autograd.py → tensor.py → ops.py
v0.2  optim/* → lr_scheduler.py
v0.3  data/dataset.py → sampler.py → dataloader.py → mnist.py
v0.4  storage.py → tensor.py → docs/TENSOR_MEMORY.md
v0.5  dispatcher.py → operator_registry.py → docs/DISPATCHER.md
v0.6  kernels.py → nn_ops.py → nn/* → docs/CNN_OPERATORS.md
v0.7  device.py → backend.py → storage.py → dispatcher.py → docs/DEVICE_BACKEND.md
v0.8  autograd.py → autograd_engine.py → tensor.py → docs/AUTOGRAD_ENGINE.md
```

v0.8 教学示例：

```bash
python examples/11_autograd_engine.py
```

完整回归：

```bash
pip install -e .
python -m unittest discover -s tests -v
```

## 6. 当前明确非目标

当前项目仍刻意不实现：

- real CUDA / CuPy runtime；
- asynchronous streams/events；
- true higher-order Autograd；
- distributed / DDP；
- eager-to-graph compiler capture；
- production kernel fusion/runtime。

这些会按照 `Plan.md` 的依赖顺序逐步进入。

## 7. 下一版：v0.9

下一阶段是 **Training Runtime / Serialization / Reproducibility**。

核心问题：

> 保存 model weights 为什么还不足以“恢复训练”？

目标状态集合：

```text
model state
optimizer state
scheduler state
RNG state
epoch / global_step
runtime metadata
      ↓
unified checkpoint
      ↓
load + validate
      ↓
deterministic resume
```

详细设计与验收任务见 `Plan.md`。