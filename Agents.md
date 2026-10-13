# Agents.md — mini-pytorch v0.13 Development Contract

> Read before modifying branch `v0.13`.

## Branch identity

- Branch: `v0.13`
- Previous: `v0.12`
- Theme: AOTAutograd-like Graph Differentiation
- Status: active until exact final-head CI passes
- Starting baseline: 192 tests

## Required architecture

```text
forward Graph
      ↓
DifferentiationSpec
      ↓
explicit saved-value ABI
      ↓
backward Graph
      ↓
GraphInterpreter
      ↓
Dispatcher
```

Rules:

- eager AutogradEngine remains the reference oracle;
- reuse the existing Graph IR rather than inventing a second compiler IR;
- VJP policy belongs in compiler differentiation rules, not backend kernels;
- numerical helper mechanisms remain backend kernels selected by Dispatcher;
- saved forward values must be explicit and deterministically ordered;
- gradient contributions to shared Values must accumulate explicitly in the backward Graph;
- CPU/EDU placement must be preserved;
- v0.13 backward execution is first-order and runs under no_grad;
- do not claim higher-order gradients;
- unsupported logical operators fail explicitly with AOTDifferentiationError;
- no lowering/fusion work belongs in this branch.

## Supported first-wave VJPs

```text
add
mul
matmul (2-D)
relu
sum
```

The implementation must handle broadcasting for add/mul and axis/keepdims semantics for sum.

## Historical invariants

All previous regression tests remain mandatory, including Storage/version safety, Dispatcher routing, Autograd Engine lifetime, deterministic checkpoints, DDP equivalence, Graph IR replay, and guarded Graph cache behavior.

## Definition of Done

Before freeze:

- explicit DifferentiationSpec;
- explicit saved forward Values;
- explicit backward Graph;
- supported VJP registry/rules;
- eager-vs-AOT forward and gradient equivalence;
- shared-subgraph accumulation test;
- CPU/EDU test;
- error-path tests;
- docs/example/README/manifest updated;
- exact final branch-head CI green;
- then freeze and integrate validated tree into main.
