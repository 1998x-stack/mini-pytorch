# MNIST training with mini-pytorch

This document shows how the existing autograd/nn/optimizer stack is extended into a complete dataset-to-training pipeline.

## What was missing

The v0.2 framework could already execute:

```text
Tensor -> Module -> loss -> backward -> optimizer -> scheduler
```

but MNIST training additionally needs:

```text
raw dataset files
  -> Dataset
  -> DataLoader / batching / shuffle
  -> image preprocessing
  -> Flatten
  -> classifier logits
  -> CrossEntropyLoss
  -> train/eval loops
  -> metrics + logging + plots + checkpoints
```

v0.3 adds those missing pieces while keeping them separate from the autograd engine.

## New components

### Data API

```text
mini_pytorch/data/
├── dataset.py      Dataset, TensorDataset
├── dataloader.py   DataLoader, default_collate
└── mnist.py        MNIST download + IDX parser
```

`DataLoader` is intentionally single-process for now. It supports batching, shuffle, deterministic seeds, `drop_last`, and custom `collate_fn`.

### Classification API

`nn.Flatten` converts `(N, 28, 28)` into `(N, 784)`.

`nn.CrossEntropyLoss` implements numerically stable softmax cross entropy for logits of shape `(N, C)` and integer class targets of shape `(N,)`. The backward pass uses the standard gradient:

```text
(softmax(logits) - one_hot(target)) / batch_size
```

for mean reduction.

## Model

The example uses a deliberately simple MLP so that the important learning path stays visible:

```text
28 x 28 image
    ↓
Flatten
    ↓
Linear(784, 128)
    ↓
ReLU
    ↓
Linear(128, 10)
    ↓
CrossEntropyLoss
```

This is enough to train a real MNIST classifier without requiring Conv2d yet.

## Install

```bash
pip install -e .
```

The v0.3 training dependencies are NumPy, Loguru, and Matplotlib.

## Fast smoke test

Start with a small subset:

```bash
python examples/06_mnist_mlp.py \
  --epochs 2 \
  --train-limit 5000 \
  --test-limit 1000 \
  --batch-size 128
```

The first run downloads the original gzip IDX resources into:

```text
data/MNIST/raw/
```

## Full dataset

```bash
python examples/06_mnist_mlp.py \
  --epochs 5 \
  --batch-size 128 \
  --hidden-size 128 \
  --lr 0.001
```

By default the full 60,000 training and 10,000 test examples are used.

## Outputs

Each run creates a timestamped directory under `runs/mnist/` containing:

```text
config.json
train.log
metrics.csv
loss_curve.png
accuracy_curve.png
best_model.npz
last_model.npz
```

Loguru writes both console progress and a persistent `train.log`. The CSV records epoch, learning rate, train/test loss, and train/test accuracy. Matplotlib regenerates the loss and accuracy curves after every epoch.

## Training loop mapping to PyTorch

The core loop remains intentionally familiar:

```python
for images, targets in train_loader:
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
```

After validation:

```python
scheduler.step()
```

The example defaults to `AdamW + CosineAnnealingLR`.

## Why this is still a mini framework

The implementation intentionally does not yet add multiprocessing workers, pinned memory, CUDA, Conv2d, BatchNorm, or automatic mixed precision. Those are separate architectural layers and can be added without changing the MNIST dataset API or the basic training loop.

A logical next step is:

```text
Sampler -> RandomSampler / SequentialSampler
DataLoader workers
Conv2d
MaxPool2d
Dropout
BatchNorm
CNN MNIST example
```
