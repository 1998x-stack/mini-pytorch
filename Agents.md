# Agents.md — mini-pytorch v0.13 Frozen Contract

> `v0.13` is a frozen historical learning snapshot. Read this before any maintenance change.

## Branch identity

- Branch: `v0.13`
- Previous: `v0.12`
- Theme: AOTAutograd-like Graph Differentiation
- Status: **frozen**
- Package: `0.13.0`
- Validated baseline: **206/206 tests passed**

## Frozen architecture

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

## Promised behavior

- eager `AutogradEngine` remains the correctness oracle;
- forward and backward reuse the same Graph IR representation;
- saved forward Values are explicit and deterministically ordered;
- shared gradient contributions accumulate through explicit backward-Graph add nodes;
- VJP policy lives in compiler differentiation rules;
- helper numerical mechanisms remain backend kernels behind Dispatcher;
- CPU/EDU placement is preserved;
- backward-Graph execution is intentionally first-order under `no_grad`;
- unsupported logical operators fail with `AOTDifferentiationError`;
- no higher-order-gradient claim is made.

## Supported VJP boundary

```text
add
mul
matmul (2-D)
relu
sum
```

Broadcasting for add/mul and axis/keepdims semantics for sum are part of the frozen contract.

## Maintenance rule

Allowed after freeze:

- documentation corrections;
- tests clarifying already-promised behavior;
- narrow bug fixes inside the v0.13 boundary.

Do not backport:

- lowering/fusion;
- new compiler stages;
- symbolic-shape systems;
- broad new VJP families;
- higher-order Autograd.

## Historical invariants

All earlier regression tests remain part of this snapshot, including Storage/version safety, Dispatcher routing, Autograd Engine lifetime, reproducible checkpoints, DDP equivalence, Graph IR replay, and guarded Graph cache behavior.

## Validation

The exact frozen branch head must remain green under:

```bash
python -m unittest discover -s tests -v
```

Frozen baseline:

```text
206 tests
OK
```

The next development work belongs on `main` → `v0.14`, not on this branch.
