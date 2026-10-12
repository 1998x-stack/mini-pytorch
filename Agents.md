# Agents.md — mini-pytorch v0.3

> Read this file before modifying this branch.

## Branch identity

- Branch: `v0.3`
- Role: active/finalizing learning snapshot
- Theme: Dataset + Sampler + BatchSampler + DataLoader + MNIST classification
- Previous version: `v0.2`
- Required companion document: `version.md`

## Mandatory reading order

1. `Agents.md`
2. `version.md`
3. `docs/ROADMAP.md`
4. relevant data/nn/training source
5. relevant tests

## Scope rule

v0.3 exists to explain the data-input side of training and complete an end-to-end classification path.

Primary architecture:

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
model
   ↓
CrossEntropyLoss
   ↓
Autograd
   ↓
Optimizer + Scheduler
```

## Allowed v0.3 work

Current/finalizing scope includes:

- Dataset/TensorDataset fixes;
- SequentialSampler/RandomSampler/BatchSampler;
- Subset and `random_split`;
- SubsetRandomSampler;
- WeightedRandomSampler;
- explicit single-process DataLoader iterator/fetcher lifecycle;
- collate behavior/tests;
- MNIST dataset/training integration;
- resumable checkpoint containing model/optimizer/scheduler/epoch state;
- confusion matrix/prediction visualization if kept small and educational.

## Explicit non-goals

Do not add these to v0.3:

- Storage/stride/view/contiguous redesign — v0.4;
- Dispatcher/operator registry — v0.5;
- Conv2d/CNN primitives beyond what current scope requires — later version;
- CUDA/device backend architecture;
- multiprocessing DataLoader workers;
- pinned memory/prefetch;
- distributed training;
- compiler stack.

Record future ideas in roadmap/planning docs rather than pulling them into this branch.

## Responsibility boundaries

- Dataset: sample access/storage semantics.
- Sampler: index selection policy.
- BatchSampler: grouping indices into batches.
- DataLoader: execution/fetch/collation lifecycle.
- `collate_fn`: convert a sample list into batch structure.
- model/loss/optimizer: must not own sampling policy.

The Dataset must not implement shuffle policy.

## Testing requirements

Data-pipeline changes should cover:

- deterministic sequential order;
- seeded random reproducibility;
- epoch-to-epoch random behavior where intended;
- replacement sampling;
- custom sampler behavior;
- `drop_last`;
- invalid option combinations;
- custom/nested collation;
- MNIST IDX parser fixtures;
- end-to-end classification smoke path.

Preferred full-suite command:

```bash
python -m unittest discover -s tests -v
```

Do not claim the whole suite passes unless run on the exact branch state.

## Integration with `main`

`main` is the canonical newest integrated state.

Any validated **code behavior** added on `v0.3` must be synchronized into `main` immediately. Do not allow v0.3 to contain newer working implementation code while main remains stale.

Branch-specific `Agents.md`/`version.md` documentation may differ from main because they describe the snapshot itself.

## Freeze rule

Before v0.3 is frozen:

- promised data pipeline is complete;
- tests pass;
- MNIST example remains coherent;
- `version.md` accurately states final delta from v0.2;
- latest validated implementation exists on `main`;
- remaining future work is moved to `Plan.md` on main.

After freeze, new architecture work must begin from latest `main`, not from `v0.3` directly.
