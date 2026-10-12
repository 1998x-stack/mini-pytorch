# mini-pytorch v0.3

## 1. Version identity

- **Branch:** `v0.3`
- **Base:** created from the latest integrated `main` state after the v0.2 optimization work
- **Theme:** data pipeline + classification + MNIST end-to-end training
- **Previous version:** `v0.2`
- **Status:** active development snapshot

v0.3 keeps the Tensor / Autograd / Module / Optimizer / Scheduler stack from v0.2 and adds the missing input-data side of a deep-learning framework.

The key architectural shift is:

```text
v0.2
Model + Optimizer + Scheduler

          ↓ v0.3 adds

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
Tensor batches
  ↓
classification model
```

---

## 2. Core difference from v0.2

### v0.2 focuses on optimization

```text
parameters
   ↓
gradients
   ↓
optimizer state
   ↓
learning-rate schedule
```

### v0.3 focuses on data flow and end-to-end training

```text
raw dataset
   ↓
index-selection policy
   ↓
batch construction
   ↓
Tensor batch
   ↓
model
   ↓
CrossEntropyLoss
   ↓
optimizer / scheduler
```

The most important new lesson is:

> A dataset stores or produces samples; it should not decide sampling order. Sampling, batching and collation are separate responsibilities.

---

## 3. Added in v0.3

### Dataset abstraction

- `Dataset`
- `TensorDataset`
- map-style `__getitem__()` / `__len__()` protocol

This provides the first explicit separation between **data representation** and **training iteration order**.

### Sampler architecture

v0.3 introduces:

- `Sampler`
- `SequentialSampler`
- `RandomSampler`
- replacement sampling
- deterministic seed behavior
- custom sampler support

Conceptually:

```text
Dataset
   │
   │ len(dataset)
   ▼
Sampler
   │
   │ produces indices
   ▼
0, 5, 9, 2, 1, ...
```

This means `shuffle=True` is now understood as a convenience choice of sampling policy rather than behavior that belongs inside Dataset.

### BatchSampler

`BatchSampler` groups individual sample indices into mini-batches:

```text
Sampler:
0, 7, 4, 2, 9, 1, ...

BatchSampler(batch_size=3):
[0, 7, 4]
[2, 9, 1]
...
```

This makes batching itself an independent index-level transformation.

### DataLoader refactor

The DataLoader is now sampler-driven:

```text
Dataset
   ↓
Sampler
   ↓
BatchSampler
   ↓
DataLoader fetch
   ↓
collate_fn
   ↓
Tensor batch
```

Supported behavior includes:

- `batch_size`
- `shuffle`
- `drop_last`
- deterministic seeded shuffle
- explicit `sampler`
- explicit `batch_sampler`
- custom `collate_fn`
- mutual-exclusion checks for incompatible arguments

### MNIST dataset

v0.3 adds direct MNIST IDX/gzip support without hiding the data format behind torchvision.

The pipeline is intentionally educational:

```text
MNIST gzip files
       ↓
IDX header parsing
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

### Classification support

The neural-network layer is expanded with:

- `Flatten`
- `CrossEntropyLoss`
- stable softmax/log-loss logic
- classification backward path

The standard MNIST model is:

```text
28 x 28 image
   ↓
Flatten
   ↓
784
   ↓
Linear(784, hidden)
   ↓
ReLU
   ↓
Linear(hidden, 10)
   ↓
logits
   ↓
CrossEntropyLoss
```

### End-to-end MNIST training example

v0.3 contains a complete training script using:

- MNIST Dataset
- DataLoader
- MLP
- CrossEntropyLoss
- AdamW
- CosineAnnealingLR
- train/eval modes
- `no_grad()` evaluation

Training outputs include:

- Loguru console logs
- `train.log`
- `metrics.csv`
- `loss_curve.png`
- `accuracy_curve.png`
- best model checkpoint
- last model checkpoint
- run configuration JSON

---

## 4. What remains unchanged from v0.2

The following subsystems are intentionally kept stable:

- Tensor numerical API
- dynamic Autograd graph
- reverse-mode backward
- Module / Parameter registration
- Linear and activation layers
- Optimizer abstraction
- SGD / Adam / AdamW / RMSprop / Adagrad / Adamax
- parameter groups
- optimizer state serialization
- LR Scheduler abstraction and scheduler implementations

That means v0.3 primarily extends the framework **horizontally** into data handling instead of rewriting the computational core.

---

## 5. Key architectural boundary

v0.3 should be read with this separation in mind:

```text
                 TRAINING LOOP
                /             \
               /               \
      DATA PIPELINE          MODEL PIPELINE
          │                        │
       Dataset                  Tensor
          │                        │
       Sampler                  Module
          │                        │
    BatchSampler               Autograd
          │                        │
      DataLoader                Loss
          │                        │
      Tensor batch             Optimizer
                                   │
                               Scheduler
```

The data subsystem should not depend on optimizer/autograd internals.

---

## 6. Important source files

Recommended reading order for the new v0.3 material:

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

Also read:

```text
docs/MNIST.md
docs/ROADMAP.md
```

---

## 7. Learning goals

By the end of v0.3, the reader should be able to explain:

1. Why Dataset and Sampler are separate abstractions.
2. Why `shuffle=True` is conceptually a sampler choice.
3. Why batching can be expressed at the index layer.
4. Why class-balanced sampling should not require rewriting a dataset.
5. Why a future DistributedSampler belongs in the sampling layer.
6. How raw MNIST IDX files become Tensor batches.
7. Why classification usually outputs logits instead of probabilities.
8. How CrossEntropyLoss connects softmax-style probabilities and class labels.
9. How a full training/evaluation/checkpoint/logging loop fits around the framework core.

---

## 8. v0.3 currently completed

Completed or already present in this branch:

- [x] Dataset
- [x] TensorDataset
- [x] Sampler base
- [x] SequentialSampler
- [x] RandomSampler
- [x] replacement sampling
- [x] deterministic seed behavior
- [x] BatchSampler
- [x] sampler-driven DataLoader
- [x] custom sampler support
- [x] custom batch-sampler support
- [x] MNIST IDX parser / loader
- [x] Flatten
- [x] CrossEntropyLoss
- [x] MNIST MLP training example
- [x] Loguru logs
- [x] metrics CSV
- [x] loss curve
- [x] accuracy curve
- [x] best / last model checkpoints
- [x] detailed roadmap

---

## 9. Planned before v0.3 is frozen

The remaining v0.3 work is mainly about making the data subsystem more complete rather than adding unrelated model features.

### Sampling utilities

- [ ] `SubsetRandomSampler`
- [ ] `WeightedRandomSampler`
- [ ] `Subset`
- [ ] `random_split`

### DataLoader lifecycle

- [ ] explicit `_SingleProcessDataLoaderIter`
- [ ] `_MapDatasetFetcher`
- [ ] clearer iterator/epoch lifecycle
- [ ] richer `collate_fn` tests

### MNIST training lifecycle

- [ ] resumable checkpoint with model + optimizer + scheduler + epoch
- [ ] confusion matrix
- [ ] prediction visualization grid

These tasks are tracked in `docs/ROADMAP.md`.

---

## 10. What v0.3 intentionally does not implement

Still out of scope:

- multiprocessing DataLoader workers
- `num_workers`
- pinned memory
- prefetch queues
- explicit Tensor Storage
- explicit stride / storage offset semantics
- true view aliasing semantics
- `contiguous()` memory materialization
- inplace version counters
- Conv2d / pooling
- CUDA/backend abstraction
- DDP

Those belong to later versions because each introduces a different mental model.

---

## 11. Version delta summary

```text
v0.2
├── Tensor / Autograd / Module
├── optimizer family
├── optimizer checkpoint state
└── LR schedulers

          ↓ v0.3 adds

v0.3
├── everything above
├── Dataset / TensorDataset
├── Sampler abstraction
├── SequentialSampler
├── RandomSampler
├── BatchSampler
├── sampler-driven DataLoader
├── MNIST IDX loader
├── Flatten
├── CrossEntropyLoss
├── end-to-end MNIST training
├── Loguru logs
├── metrics CSV
├── training curves
└── model checkpoints
```

The short distinction is:

> **v0.2 teaches how training state is optimized over time. v0.3 teaches how data enters that training system.**

---

## 12. Next version

The planned `v0.4` moves into Tensor memory semantics:

```text
Storage
  ↓
shape + stride + storage_offset
  ↓
transpose / permute
  ↓
contiguous vs non-contiguous
  ↓
view vs reshape
  ↓
contiguous()
  ↓
inplace mutation
  ↓
version counter / autograd safety
```

This is intentionally postponed until the end-to-end training/data abstractions are stable.

For the detailed plan, read `docs/ROADMAP.md` on this branch.
