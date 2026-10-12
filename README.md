# mini-pytorch v0.3 — Dataset / Sampler / DataLoader + MNIST Classification

> 在已有 Tensor / Autograd / Optimizer 之上，补齐训练系统的“数据输入侧”。

**版本主题：** Dataset → Sampler → BatchSampler → DataLoader → classification → MNIST

**状态：** historical learning snapshot

## 1. v0.3 相比 v0.2 的核心变化

v0.2 重点研究：

```text
参数如何更新？
```

v0.3 转向：

```text
训练数据如何被组织、采样、批处理并送入模型？
```

核心链路：

```text
raw data
   ↓
Dataset
   ↓
Sampler
   ↓
BatchSampler
   ↓
DataLoader
   ↓
collate_fn
   ↓
Tensor batch
   ↓
Model
   ↓
CrossEntropyLoss
   ↓
Optimizer / Scheduler
```

最重要的架构边界：

> Dataset 负责“样本是什么”，Sampler 负责“按什么顺序取样本”，BatchSampler 负责“哪些 index 组成一个 batch”。

## 2. Data subsystem

v0.3 引入：

- `Dataset`
- `TensorDataset`
- `Sampler`
- `SequentialSampler`
- `RandomSampler`
- replacement sampling
- deterministic seed
- `BatchSampler`
- sampler-driven `DataLoader`
- custom sampler
- custom batch sampler
- custom `collate_fn`

因此：

```python
DataLoader(dataset, shuffle=True)
```

本质上可以理解为：

```text
shuffle=True
   ↓
选择 RandomSampler policy
```

而不是 Dataset 自己负责 shuffle。

## 3. MNIST pipeline

这一版没有依赖 torchvision 隐藏数据格式，而是直接实现 IDX/gzip 读取：

```text
MNIST gzip
   ↓
IDX header parser
   ↓
NumPy arrays
   ↓
MNIST Dataset
   ↓
Sampler / BatchSampler
   ↓
DataLoader
   ↓
Tensor batches
```

配套分类组件包括：

- `Flatten`
- `CrossEntropyLoss`
- stable softmax-style loss computation
- classification backward

典型模型：

```text
28 x 28
  ↓
Flatten
  ↓
784
  ↓
Linear
  ↓
ReLU
  ↓
Linear(10)
  ↓
logits
  ↓
CrossEntropyLoss
```

## 4. 完整 MNIST training lifecycle

示例包含：

- AdamW
- CosineAnnealingLR
- train/eval mode
- `no_grad()` evaluation
- Loguru logs
- `metrics.csv`
- loss curve
- accuracy curve
- best checkpoint
- last checkpoint

这意味着 mini-pytorch 从“能训练一个 toy example”进入了“能组织完整 supervised training loop”的阶段。

## 5. 推荐阅读顺序

```text
mini_pytorch/data/dataset.py
        ↓
mini_pytorch/data/sampler.py
        ↓
mini_pytorch/data/dataloader.py
        ↓
mini_pytorch/data/mnist.py
        ↓
mini_pytorch/nn/flatten.py
        ↓
mini_pytorch/nn/loss.py
        ↓
examples/06_mnist_mlp.py
```

重点理解：

1. Dataset 为什么不负责 shuffle；
2. Sampler 为什么输出 index 而不是 sample；
3. BatchSampler 为什么属于 index layer；
4. DataLoader 为什么负责 fetch + collate；
5. 分类模型为什么输出 logits；
6. CrossEntropyLoss 如何产生 logits gradient。

## 6. 快速运行

```bash
pip install -e .
python -m unittest discover -s tests -v
python examples/06_mnist_mlp.py --help
```

完整 MNIST 运行方式和数据说明见：

```text
docs/MNIST.md
```

## 7. v0.3 保持不变的部分

从 v0.2 继承：

- Tensor / dynamic Autograd
- Module / Parameter
- Linear / activation
- Optimizer family
- optimizer state serialization
- parameter groups
- LR Scheduler subsystem

本版本主要是横向补齐 data pipeline，而不是重写计算核心。

## 8. 本版本不做什么

v0.3 暂不进入：

- multiprocessing DataLoader
- `num_workers`
- pinned memory / prefetch
- explicit Storage
- stride / storage offset
- real view alias semantics
- `contiguous()`
- inplace version counter
- Dispatcher / backend routing
- Conv2d / CUDA / DDP / compiler

## 9. 从这里最值得带走的认知

```text
Dataset != DataLoader
Sampler != Dataset
BatchSampler != collate_fn
```

它们之所以被拆开，是因为数据存储、索引策略、批次策略、样本拼装是四个不同的职责。

## 10. 下一版本

`v0.4` 深入 Tensor 本身：

```text
Storage
  ↓
shape + stride + storage_offset
  ↓
view / transpose / slicing
  ↓
contiguous vs non-contiguous
  ↓
view vs reshape
  ↓
inplace + version counter
```

详细版本边界见 [`version.md`](./version.md)。