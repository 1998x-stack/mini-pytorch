# Agents.md — mini-pytorch v0.1

> Read this file before modifying this branch.

## Branch identity

- Branch: `v0.1`
- Role: frozen historical learning snapshot
- Theme: Tensor + dynamic Autograd + Module + basic Optimizer MVP
- Required companion document: `version.md`

## Mandatory rule

This branch exists to preserve the first complete mini-pytorch mental model. Do not add later-version subsystems here.

Allowed changes after freeze:

- documentation corrections;
- test corrections that preserve v0.1 semantics;
- narrowly scoped bug fixes required for behavior already promised by `version.md`.

Forbidden here:

- LR scheduler suite from v0.2;
- Dataset/DataLoader/Sampler from v0.3;
- Storage/stride/view redesign from v0.4;
- Dispatcher/backend/compiler/distributed features from later versions.

## Required reading order

1. `Agents.md`
2. `version.md`
3. relevant source
4. relevant tests

## Project principles

mini-pytorch is educational. Prefer:

```text
correctness
  ↓
readability
  ↓
explicit invariants
  ↓
tests
  ↓
optimization
```

Use real PyTorch only as an architectural reference. Do not copy upstream implementation code.

## v0.1 architecture boundary

```text
Tensor
  ↓
Operator
  ↓
Dynamic graph
  ↓
Autograd / backward
  ↓
Parameter
  ↓
Module
  ↓
Loss
  ↓
Optimizer
```

Keep responsibilities separated:

- Tensor/ops construct computation;
- Autograd owns backward graph traversal;
- Module owns Parameters/submodules;
- Optimizer owns parameter-update policy/state.

## Testing rule

Before reporting success, run applicable tests on the exact branch state. Do not claim a full suite passed unless it actually ran successfully.

Preferred command:

```bash
python -m unittest discover -s tests -v
```

For autograd behavior, prefer numerical/invariant checks rather than only training convergence.

## Documentation rule

If a bug fix changes promised v0.1 behavior, update `version.md` so the historical snapshot remains self-explanatory.

## Main-branch rule

`main` is the canonical newest integrated state. Do not use `v0.1` as the base for new future versions. New versions must branch from latest `main`.
