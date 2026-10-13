# Agents.md — mini-pytorch v0.3

> Read this file before modifying branch `v0.3`.

## Branch identity

- Branch: `v0.3`
- Role: **frozen historical learning snapshot**
- Theme: Dataset + Sampler + BatchSampler + DataLoader + MNIST classification
- Previous version: `v0.2`
- Companion document: `version.md`
- Current validation baseline: **48 tests / OK**

## Mandatory reading order

1. `Agents.md`
2. `version.md`
3. relevant source
4. relevant tests
5. historical roadmap only as context

## Frozen scope

This branch teaches the data-input side of training:

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
   ↓
model / CrossEntropyLoss
```

The implemented snapshot includes Dataset/TensorDataset, SequentialSampler, RandomSampler, replacement sampling, BatchSampler, sampler-driven DataLoader, MNIST loading, Flatten, CrossEntropyLoss and the MNIST MLP example.

## Allowed maintenance

Only narrow maintenance is allowed:

- documentation corrections describing this snapshot;
- tests that verify behavior already promised by v0.3;
- CI/tooling maintenance;
- narrow correctness fixes strictly inside existing v0.3 behavior when the historical semantics are demonstrably wrong.

Any such code fix must not import later architectural mechanisms merely because they exist on `main`.

## Former planned work is closed

The active-development roadmap once considered:

- `Subset` / `random_split`;
- `SubsetRandomSampler`;
- `WeightedRandomSampler`;
- explicit single-process iterator/fetcher internals;
- resumable optimizer/scheduler/epoch checkpoints;
- confusion-matrix / prediction visualization helpers.

Those items are **not open work on v0.3** and are not part of its frozen completion contract. Some concepts were later assigned to different version layers (for example, resumable training state in v0.9).

## Explicit non-goals

Do not add later mechanisms to this branch:

- Storage / stride / view / contiguous redesign;
- Dispatcher / operator registry;
- CNN operator stack;
- device/backend architecture;
- multiprocessing/prefetch/pinned-memory DataLoader;
- runtime checkpoint subsystem from v0.9;
- distributed training;
- compiler stack.

## Responsibility boundaries

- Dataset: sample access/storage semantics.
- Sampler: index-selection policy.
- BatchSampler: grouping indices into batches.
- DataLoader: fetch/collation lifecycle.
- `collate_fn`: converts sample lists into batch structure.
- model/loss/optimizer: must not own sampling policy.

## Testing contract

For any permitted maintenance, run:

```bash
python -m unittest discover -s tests -v
```

Do not claim branch health without exact-head CI evidence.

## Relationship to main

`main` is the canonical newest integrated state. v0.3 is intentionally older and must never become the base for new architecture work.

```text
historical study / narrow correction
           ↑
          v0.3

new framework development
           ↑
          main
```

Do not reopen the historical roadmap from this branch.
