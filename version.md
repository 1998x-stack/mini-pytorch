# mini-pytorch v0.3

## Version identity

- **Branch:** `v0.3`
- **Previous version:** `v0.2`
- **Theme:** Dataset + Sampler + BatchSampler + DataLoader + MNIST classification
- **Status:** frozen historical learning snapshot
- **Package version:** `0.3.0`
- **Current exact-head validation:** **48 / 48 tests passed** on Python 3.11

v0.3 records the point where mini-pytorch first gained a coherent data-input path and an end-to-end classification example. It is no longer an active development branch. New framework work must start from latest `main`.

## Core delta from v0.2

```text
v0.2
Tensor / Autograd / Module
        ↓
Optimizer / Scheduler

v0.3 adds

Dataset
   ↓
Sampler
   ↓
BatchSampler
   ↓
DataLoader
   ↓
Tensor batches
   ↓
classification model
   ↓
CrossEntropyLoss
```

The central lesson is:

> Dataset owns sample access; Sampler owns index-selection policy; BatchSampler owns grouping; DataLoader owns fetching/collation.

## Added in this frozen version

### Data abstractions

- `Dataset`
- `TensorDataset`
- `Sampler`
- `SequentialSampler`
- `RandomSampler`
- replacement sampling
- deterministic seeded sampling behavior
- `BatchSampler`
- sampler-driven `DataLoader`
- custom sampler / batch-sampler support
- custom `collate_fn`
- argument-conflict validation

### Classification path

- MNIST IDX/gzip parsing and dataset loading
- `Flatten`
- `CrossEntropyLoss`
- MLP MNIST training example
- train/eval usage
- `no_grad()` evaluation
- Loguru logs
- metrics CSV
- loss/accuracy curves
- best / last model checkpoints
- run configuration JSON

## Architecture boundary

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

The data subsystem does not own optimizer or autograd policy.

## Important source files

```text
mini_pytorch/data/dataset.py
mini_pytorch/data/sampler.py
mini_pytorch/data/dataloader.py
mini_pytorch/data/mnist.py
mini_pytorch/nn/container.py
mini_pytorch/nn/loss.py
examples/06_mnist_mlp.py
```

## Learning goals

A reader should be able to explain:

1. why Dataset and Sampler are separate abstractions;
2. why `shuffle=True` is a sampling policy choice;
3. why batching can happen at the index layer;
4. how raw MNIST IDX data becomes Tensor batches;
5. why classification models emit logits;
6. how CrossEntropyLoss fits into the training loop;
7. how the data path composes with the v0.2 optimizer/scheduler stack.

## Frozen completion set

Implemented in this branch:

- [x] Dataset / TensorDataset
- [x] Sampler base
- [x] SequentialSampler
- [x] RandomSampler
- [x] replacement sampling
- [x] deterministic seed behavior
- [x] BatchSampler
- [x] sampler-driven DataLoader
- [x] custom sampler support
- [x] custom batch-sampler support
- [x] MNIST IDX loader
- [x] Flatten
- [x] CrossEntropyLoss
- [x] end-to-end MNIST MLP example
- [x] logging / metrics / plots / simple model checkpoints
- [x] exact-head CI validation

## Former planned items that are not part of frozen v0.3

During active development the roadmap considered the following additions, but they were **not implemented in the frozen v0.3 snapshot**:

- `SubsetRandomSampler`
- `WeightedRandomSampler`
- `Subset`
- `random_split`
- explicit `_SingleProcessDataLoaderIter`
- explicit `_MapDatasetFetcher`
- richer collation lifecycle
- resumable model + optimizer + scheduler + epoch checkpointing
- confusion matrix / prediction visualization

These entries are historical planning context, not promises that remain open on this branch. In particular, deterministic resumable checkpointing later became a v0.9 runtime concern rather than being backported into v0.3.

## Intentional non-goals

This version does not implement:

- multiprocessing DataLoader workers / `num_workers`
- pinned memory / prefetch queues
- explicit Storage / stride / storage offset semantics
- true view aliasing / `contiguous()` semantics
- inplace version counters
- Dispatcher / backend architecture
- CNN operator stack
- distributed training
- compiler/runtime stack

Those mechanisms belong to later version snapshots.

## Validation

Current branch validation command:

```bash
python -m unittest discover -s tests -v
```

The current v0.3 HEAD was revalidated in the 2026 review environment with:

```text
Python 3.11
48 tests
OK
```

## Next historical milestone

The next learning snapshot is `v0.4`, which introduces Tensor memory semantics:

```text
Storage
  ↓
shape + stride + storage_offset
  ↓
views / transpose / permute
  ↓
contiguous vs non-contiguous
  ↓
inplace mutation / version safety
```

Do not continue feature development from v0.3. Use latest `main` for new work.
