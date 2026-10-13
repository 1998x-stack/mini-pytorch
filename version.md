# mini-pytorch v0.11

## 1. Version identity

- **Branch:** `v0.11`
- **Previous version:** `v0.10`
- **Theme:** Graph IR / Eager-to-Graph Capture
- **Status:** active development
- **Target package version:** `0.11.0`

v0.11 answers:

> How can eager logical-operator execution be represented as explicit `Value / Node / Graph` data and then replayed through the existing Dispatcher without creating a second numerical framework?

## 2. Core delta from v0.10

v0.10 made replicated gradient synchronization explicit:

```text
local backward → Reducer → ProcessGroup → synchronized grad
```

v0.11 introduces the first compiler-facing representation:

```text
Python Tensor calls
      ↓
Dispatcher
      ├── eager execution
      └── capture observer
              ↓
        Value / Node / Graph
              ↓
       GraphInterpreter
              ↓
          Dispatcher
              ↓
       ordinary eager execution
```

The graph records logical operator semantics. It does not own kernels, Autograd formulas, backend selection, optimization or code generation.

## 3. Planned mechanisms

- explicit `Value`, `Node`, `Graph` IR;
- graph input and Tensor/Python constant representation;
- capture context with nested-capture rejection;
- Dispatcher-level observer that records only outer logical dispatch, not redispatch internals;
- deterministic value/node IDs and stable textual IR;
- supported operator tracing for the existing Dispatcher operator set;
- GraphInterpreter replay through the existing Dispatcher;
- CPU/EDU placement preservation via replay inputs/constants;
- eager-vs-graph forward equivalence;
- eager-vs-graph input-gradient equivalence;
- explicit `GraphBreakError` for unsupported eager dataflow rather than silent miscapture.

## 4. Non-goals

v0.11 does **not** implement:

- Python bytecode tracing;
- symbolic Python control flow;
- shape guards or graph cache;
- Dynamo frame evaluation;
- AOTAutograd;
- graph optimization/fusion;
- backend code generation;
- Inductor/Triton;
- full Module/Parameter lifting;
- arbitrary mutation capture.

Those belong to later milestones.

## 5. Required correctness proof

For a supported function built from captured logical operators:

```text
same fresh inputs
      ↓
 eager function ───────┐
                       ├── outputs equal
 captured Graph replay ┘

backward(eager output)
backward(replayed output)
      ↓
input gradients equal
```

Replay must exercise the normal Dispatcher and Autograd stack.

## 6. Definition of Done

Before freeze:

- compiler IR/capture/interpreter implementation complete;
- error semantics covered;
- stable textual IR documented;
- full regression green;
- version README/docs/example complete;
- exact final branch-head GitHub Actions green;
- branch frozen before integration to `main`.
