# mini-pytorch Roadmap

> Goal: use a small, readable NumPy-based framework to learn the architectural ideas behind real PyTorch, one subsystem at a time.

This roadmap is intentionally organized by **learning boundary**, not by feature count. Each version should introduce one new mental model while keeping previous layers stable.

---

## 0. Branch / version map

The repository default branch is `main` (there is no `master` branch). Version branches are historical learning snapshots:

| Branch | Historical base | Theme | Status |
|---|---|---|---|
| `v0.1` | `00b544a85c0c48d92a47b00efdc299678244e6c3` | Tensor + Autograd + Module + basic Optimizer MVP | frozen learning snapshot |
| `v0.2` | `ed30650a2e3c9d913244e85ddd508000f62ee363` | optimizer suite + LR schedulers + checkpointable optimizer state | frozen learning snapshot |
| `v0.3` | based on current `main` | Dataset / Sampler / DataLoader + MNIST end-to-end training | active development |
| `v0.4` | planned from completed `v0.3` | Storage / stride / view / contiguous / reshape | planned |
| `v0.5` | planned | Conv2d / Pooling / CNN training | planned |
| `v0.6` | planned | device/backend abstraction | planned |
| `v0.7` | planned | advanced autograd | planned |
| `v0.8` | planned | PyTorch internals mapping | planned |

Branch policy:

- `v0.1` and `v0.2` are historical snapshots and should not receive normal feature work.
- active work for a version happens on that version branch.
- `main` represents the latest integrated state.
- every version must have tests, examples, and architecture documentation before being considered complete.

---

# v0.3 — Data pipeline and end-to-end classification

## Learning objective

Understand why PyTorch splits the input pipeline into several independent concepts:

```text
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
```

The important lesson is that **loading data and choosing indices are different responsibilities**.

## Current baseline already available

- `Dataset`
- `TensorDataset`
- single-process `DataLoader`
- MNIST IDX download / parsing
- `Flatten`
- `CrossEntropyLoss`
- MNIST MLP example
- Loguru training logs
- metrics CSV
- loss / accuracy plots
- AdamW + CosineAnnealingLR training path

## v0.3.1 — Sampler architecture

### Required implementation

- [x] abstract `Sampler`
- [x] `SequentialSampler`
- [x] `RandomSampler`
- [x] replacement sampling
- [x] deterministic seed behavior
- [x] `BatchSampler`
- [x] make `DataLoader` sampler-driven
- [x] `shuffle=True` becomes convenience syntax for `RandomSampler`
- [x] custom sampler support
- [x] custom batch sampler support
- [x] mutual-exclusion validation (`sampler` vs `shuffle`, etc.)

### Core learning questions

By the end of this stage, the reader should be able to explain:

1. Why `Dataset.__getitem__()` should not decide shuffle order.
2. Why `shuffle=True` is conceptually a sampler choice.
3. Why batching can be expressed as a sampler-of-indices problem.
4. Why class-balanced sampling should not modify the dataset.
5. Why distributed training eventually needs a `DistributedSampler` rather than a special Dataset.

### Acceptance criteria

- sequential sampling exactly reproduces dataset order;
- seeded random samplers are reproducible across instances;
- one sampler produces different permutations across epochs;
- replacement sampling supports `num_samples`;
- `BatchSampler` handles `drop_last` correctly;
- DataLoader still passes all existing batching / shuffle tests;
- a user-defined sampler changes sample order without changing Dataset code.

---

## v0.3.2 — Sampling extensions

These are deliberately optional after the core sampler model is stable.

### Planned

- [ ] `SubsetRandomSampler`
- [ ] `WeightedRandomSampler`
- [ ] `Subset`
- [ ] `random_split`
- [ ] simple train/validation split example

### Why this matters

`WeightedRandomSampler` provides a clean demonstration of handling class imbalance at the **index-selection layer**.

Example target API:

```python
weights = np.array([...])
sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
loader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

---

## v0.3.3 — DataLoader lifecycle

### Planned

- [ ] explicit DataLoader iterator class (`_SingleProcessDataLoaderIter`)
- [ ] separate fetcher object (`_MapDatasetFetcher`)
- [ ] clearer iterator lifecycle / epoch boundaries
- [ ] custom `collate_fn` example
- [ ] nested tuple/list collation tests
- [ ] error-path tests for malformed batches

This mirrors the way real PyTorch keeps DataLoader configuration separate from iterator execution.

### Not in v0.3

Do **not** add multiprocessing yet:

- no `num_workers` implementation;
- no worker subprocesses;
- no queues;
- no prefetch;
- no pinned memory.

Those mechanisms are much easier to understand after the single-process iterator/fetcher model is explicit.

---

## v0.3.4 — MNIST as integration test

MNIST should remain the first end-to-end classification benchmark.

Training path:

```text
MNIST IDX gzip files
        ↓
MNIST Dataset
        ↓
RandomSampler
        ↓
BatchSampler
        ↓
DataLoader
        ↓
Flatten
        ↓
Linear(784, hidden)
        ↓
ReLU
        ↓
Linear(hidden, 10)
        ↓
CrossEntropyLoss
        ↓
Autograd
        ↓
AdamW
        ↓
CosineAnnealingLR
```

### Required outputs

- [x] Loguru console/file logs
- [x] `metrics.csv`
- [x] `loss_curve.png`
- [x] `accuracy_curve.png`
- [x] best model checkpoint
- [x] last model checkpoint
- [ ] resume-training checkpoint containing model + optimizer + scheduler + epoch
- [ ] confusion matrix utility
- [ ] prediction visualization for a small sample grid

### v0.3 Definition of Done

v0.3 is complete when:

1. Dataset / Sampler / BatchSampler / DataLoader boundaries are explicit.
2. MNIST training uses those abstractions without custom data-loop hacks.
3. CrossEntropy backward passes numerical checks.
4. all unit tests pass.
5. the MNIST smoke-test command is documented.
6. `v0.3` can be read independently as a coherent learning snapshot.

---

# v0.4 — Tensor memory model: Storage, stride, view, contiguous

## Why v0.4 comes next

The current Tensor largely delegates shape semantics to NumPy. That is enough for autograd, but it hides one of the most important PyTorch concepts:

> A Tensor is not simply "an n-dimensional array". It is a view over storage described by shape, stride, offset and dtype.

This version should make that model explicit.

## Target mental model

```text
Storage: [a b c d e f ...]
          ↑
          └── storage_offset

Tensor metadata
├── shape
├── stride
├── dtype
├── storage_offset
└── storage reference
```

## v0.4.1 — Storage abstraction

### Tasks

- [ ] introduce `Storage`
- [ ] Tensor owns/references Storage rather than treating ndarray as the whole identity
- [ ] expose `storage_offset`
- [ ] expose shape + stride metadata
- [ ] define contiguous stride calculation

### Tests

- [ ] same Storage can back multiple Tensor views;
- [ ] mutating shared storage is visible from each view (before autograd safety rules);
- [ ] contiguous stride generation is correct for 1-D/2-D/3-D cases.

---

## v0.4.2 — transpose and non-contiguous tensors

### Tasks

- [ ] general `transpose(dim0, dim1)`
- [ ] `permute(*dims)`
- [ ] `is_contiguous()`
- [ ] preserve storage for metadata-only transpose

### Learning target

Explain this classic PyTorch behavior:

```python
x = ...
y = x.transpose(0, 1)

# same underlying data, different stride
```

---

## v0.4.3 — view vs reshape

### Tasks

- [ ] implement `view()` with stride compatibility checks
- [ ] implement `reshape()` as "view when possible, copy when necessary"
- [ ] implement `contiguous()`
- [ ] expose whether an operation returns a view or copy

### Required demonstrations

```text
contiguous tensor -> view succeeds without copy
transpose -> non-contiguous tensor
non-contiguous -> view may fail
non-contiguous -> reshape may copy
contiguous() -> materializes contiguous layout
```

### Core questions to answer

1. Why can `view()` fail while `reshape()` works?
2. Why does transpose often produce a non-contiguous Tensor?
3. Why is `contiguous()` sometimes required before a view?
4. Why can two tensors with different shapes share the same storage?

---

## v0.4.4 — slicing / indexing views

### Planned

- [ ] basic slice indexing
- [ ] stride/offset-only slice where possible
- [ ] negative / stepped slices where reasonable
- [ ] backward through views

Avoid full NumPy indexing semantics in the first implementation; prioritize readable stride logic.

---

## v0.4.5 — inplace safety and version counter

Once views share storage, inplace mutation becomes an autograd correctness problem.

### Tasks

- [ ] Tensor version counter
- [ ] version bumps on inplace mutation
- [ ] saved-tensor version checks in autograd
- [ ] educational inplace operation(s)
- [ ] error when backward depends on a tensor modified after forward

This stage should directly explain why real PyTorch sometimes reports:

```text
one of the variables needed for gradient computation has been modified by an inplace operation
```

### v0.4 Definition of Done

- Storage is explicit.
- stride is explicit.
- transpose/permute can create non-contiguous views.
- `view`, `reshape`, and `contiguous` exhibit meaningfully different behavior.
- shared-storage behavior is demonstrated by tests.
- inplace version-counter checks protect autograd.

---

# v0.5 — CNN primitives

After tensor memory semantics are understood, add image operators.

## Tasks

- [ ] `Conv2d` forward
- [ ] Conv2d backward: input / weight / bias gradients
- [ ] first readable implementation using loops
- [ ] optimized educational implementation using im2col
- [ ] `MaxPool2d`
- [ ] Dropout
- [ ] BatchNorm1d / BatchNorm2d
- [ ] CNN MNIST example

## Learning questions

- Why is convolution parameter sharing?
- What exactly is a convolution weight gradient?
- Why can im2col turn convolution into matrix multiplication?
- Why does BatchNorm have parameters and buffers?
- Why does Dropout depend on `train()` / `eval()`?

---

# v0.6 — Device / backend abstraction

The core framework currently assumes NumPy / CPU.

## Planned architecture

```text
Tensor
  ↓
Backend API
  ├── NumPyBackend (CPU)
  └── optional CuPyBackend (GPU)
```

### Tasks

- [ ] `Device` abstraction
- [ ] `.to(device)`
- [ ] backend-owned allocation
- [ ] backend operator primitives
- [ ] CPU as reference backend
- [ ] optional CuPy experiment

Do not attempt a CUDA C++ backend at this stage.

---

# v0.7 — Advanced autograd

## Tasks

- [ ] `retain_grad()` for non-leaf tensors
- [ ] graph lifetime model
- [ ] `retain_graph`
- [ ] gradient hooks
- [ ] `requires_grad_()`
- [ ] `detach_()` discussion / constraints
- [ ] optional `create_graph`
- [ ] second-order gradient demonstration

## Learning target

Move from "backward works" to understanding **autograd graph lifetime and gradient ownership**.

---

# v0.8 — Map mini-pytorch to real PyTorch internals

This version is mostly architectural documentation plus small experiments.

## Topics

- Python `torch.Tensor` vs C++ TensorImpl
- Storage / TensorImpl
- Dispatcher
- ATen operators
- Autograd-generated backward nodes
- CPU/CUDA dispatch keys
- caching allocator
- streams
- DataLoader multiprocessing
- DDP + AllReduce
- `torch.compile`
- TorchDynamo
- AOTAutograd
- Inductor
- Triton

Target document:

```text
mini-pytorch concept
        ↓
real PyTorch Python API
        ↓
real PyTorch C++ subsystem
```

---

# Cross-version engineering rules

## 1. Keep the dependency direction simple

```text
Tensor / Autograd
      ↑
     nn
      ↑
 training code

Dataset / Sampler / DataLoader
      ↓
 training code
```

The data system must not depend on optimizer/autograd internals.

## 2. Every abstraction needs a minimal test

A new subsystem is incomplete until it has:

- happy-path tests;
- shape/state tests;
- at least one error-path test;
- one integration example when appropriate.

## 3. Prefer readable reference implementations first

Optimization sequence:

```text
correct + readable
      ↓
well-tested
      ↓
profiled
      ↓
optimized implementation
```

Never hide the core algorithm merely to make the mini framework faster.

## 4. Keep public APIs PyTorch-like when useful, not blindly identical

The project should copy the *conceptual boundary* of PyTorch. Exact edge-case parity is optional unless it teaches an important concept.

## 5. Version completion checklist

Before freezing a new version branch:

- [ ] all tests pass;
- [ ] package imports cleanly;
- [ ] examples run;
- [ ] README / roadmap updated;
- [ ] architecture doc updated;
- [ ] version number updated;
- [ ] no generated datasets / run artifacts committed;
- [ ] historical branch points documented.

---

# Recommended learning order

```text
v0.1  Autograd + Module + Optimizer basics
  ↓
v0.2  Optimizer state + schedulers
  ↓
v0.3  Dataset + Sampler + DataLoader + MNIST
  ↓
v0.4  Storage + stride + views + contiguous
  ↓
v0.5  Conv2d + CNN
  ↓
v0.6  devices / backends
  ↓
v0.7  advanced autograd
  ↓
v0.8  map concepts to real PyTorch internals
```

The intended outcome is not simply a larger framework. Each version should remove one major piece of "PyTorch magic" by implementing the mechanism in readable Python.
