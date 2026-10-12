# Agents.md — mini-pytorch v0.2

> Read this file before modifying this branch.

## Branch identity

- Branch: `v0.2`
- Role: frozen historical learning snapshot
- Theme: optimizer architecture + optimizer state + learning-rate scheduling
- Previous version: `v0.1`
- Required companion document: `version.md`

## Mandatory rule

This branch preserves the optimization-layer milestone. Do not add future-version features here.

Allowed changes after freeze:

- documentation corrections;
- test corrections that preserve v0.2 semantics;
- narrowly scoped fixes to optimizer/scheduler behavior promised by `version.md`.

Forbidden here:

- Dataset/DataLoader/Sampler/MNIST data-system work from v0.3;
- Storage/stride/view redesign from v0.4;
- Dispatcher, backend, distributed or compiler systems from later versions.

## Required reading order

1. `Agents.md`
2. `version.md`
3. optimizer/scheduler source
4. related tests

## v0.2 mental model

```text
Parameter
   │
   ├──────────► gradient
   │
Optimizer
   ├──────────► parameter groups
   └──────────► per-parameter state
                         ▲
                         │
                    Scheduler
                    mutates LR
```

Keep these responsibilities separate:

- Autograd computes gradients;
- Optimizer owns update rules and optimizer state;
- Scheduler owns learning-rate evolution;
- Module owns model parameters/buffers.

## Engineering rules

Prefer conceptual PyTorch compatibility, not exact edge-case parity. New optimizer algorithms should fit the existing Optimizer abstraction rather than modify Tensor/Autograd.

Tests should cover:

- mathematical update behavior;
- convergence where useful;
- parameter groups;
- state_dict/load_state_dict round trips;
- scheduler progression;
- scheduler state restoration;
- invalid configuration errors.

Preferred full-suite command:

```bash
python -m unittest discover -s tests -v
```

Do not report a full pass unless the exact branch state was tested.

## Documentation rule

Any correction that changes v0.2’s promised behavior must update `version.md`.

## Main-branch rule

`main` is the canonical newest integrated state. Do not use `v0.2` as the base for future version development. New versions must branch from latest `main`.
