# Agents.md — mini-pytorch v0.11 Frozen Branch Contract

> Read this file before modifying frozen branch `v0.11`.

## Branch identity

- Branch: `v0.11`
- Previous version: `v0.10`
- Primary mental model: **Graph IR / Eager-to-Graph Capture**
- Status: **frozen historical learning snapshot**
- Package version: `0.11.0`
- Validated baseline: **176/176 tests passed**

## 1. Frozen architecture

```text
Python Tensor API
      ↓
Dispatcher.dispatch
      ├── ordinary eager execution
      └── capture observer
              ↓
       Value / Node / Graph
              ↓
       GraphInterpreter
              ↓
          Dispatcher
              ↓
Autograd / NoGrad / CPU / EDU
```

The Graph records logical operator semantics. It is not a second numerical runtime.

## 2. Historical branch rule

Allowed only:

- documentation corrections;
- tests clarifying promised v0.11 behavior;
- narrowly scoped correctness fixes inside Graph IR/capture/interpreter semantics.

Forbidden:

- v0.12 guards/cache/recompile;
- AOTAutograd;
- fusion/lowering/codegen;
- unrelated distributed/backend/NN expansion;
- backporting later compiler milestones.

## 3. Frozen capture invariant

- capture observes only outer `Dispatcher.dispatch()` calls;
- internal `redispatch()` calls are not nodes;
- one logical eager op maps to one Node;
- capture executes normal eager math and records after execution;
- graph inputs are explicit positional Tensor inputs;
- pre-existing leaf Tensor constants may be captured by value;
- unsupported dynamic non-leaf dataflow raises `GraphBreakError`;
- nested active capture is rejected.

Do not weaken GraphBreak into silent constant freezing.

## 4. Frozen IR invariant

Core types:

```text
Value
ValueRef
TensorConstant
Node
Graph
```

Value kinds:

```text
input
constant
intermediate
```

Graph Value/Node ids must remain deterministic within capture order.

Runtime Python object identity may be used only transiently while mapping eager Tensors to IR Values; persistent graph references use Value ids.

## 5. Frozen interpreter invariant

`GraphInterpreter` must:

```text
bind runtime inputs
      ↓
materialize graph constants
      ↓
resolve structured args/kwargs
      ↓
Dispatcher.dispatch(node.operator, ...)
      ↓
bind outputs
      ↓
return graph outputs
```

It must not call backend kernels directly.

Replay must therefore preserve runtime Autograd/NoGrad/backend dispatch policy.

## 6. Dispatch-policy invariant

The Graph stores logical operators, not trace-time DispatchKeys.

A trace under NoGrad may replay under Autograd if runtime state requires it.

Do not persist:

```text
selected CPU/EDU key
selected Autograd/NoGrad key
kernel function object
```

as the semantic Node identity.

## 7. Constant invariant

Tensor constants are captured by value and materialized fresh during replay.

Captured device is retained in v0.11. Mixed-device semantics continue to be enforced by the existing Dispatcher.

Full Module/Parameter lifting, mutation analysis and device specialization are outside this frozen version.

## 8. GraphBreak invariant

An untracked Tensor with a GradFn indicates dynamic dataflow produced by an unsupported eager path.

Examples include current direct `Function.apply` operations such as `exp()`/`log()` when their outputs cross into or become outputs of a captured graph.

Such cases must raise `GraphBreakError` rather than be silently baked into constants.

## 9. Correctness proof that must remain

The full suite proves:

```text
eager forward == graph replay forward
```

and for supported differentiable graphs:

```text
eager input grads == replay input grads
```

Additional required regression coverage includes:

- stable textual IR;
- Tensor constant copy semantics;
- kwargs capture;
- matmul replay;
- conv2d replay;
- EDU replay;
- trace NoGrad → replay Autograd;
- replay uses Dispatcher;
- wrong input count;
- nested capture rejection;
- GraphBreak semantics.

## 10. Historical baseline

Validated implementation suite:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
Ran 176 tests
OK
```

Exact frozen branch-head CI must remain green.

## 11. Documentation map

```text
README.md
version.md
mini_pytorch/compiler/ir.py
mini_pytorch/compiler/capture.py
mini_pytorch/compiler/interpreter.py
mini_pytorch/dispatcher.py
tests/test_graph_ir.py
docs/GRAPH_IR.md
examples/14_graph_capture.py
```

## 12. Explicit non-goals

v0.11 does not claim:

- Python frame/bytecode capture;
- control-flow tracing;
- shape guards;
- graph cache;
- automatic recompilation;
- symbolic shapes;
- AOTAutograd;
- optimization/fusion;
- lowering/codegen;
- Inductor/Triton;
- production compiler performance.

## 13. Next-version boundary

`v0.12` must be created from latest validated `main` and owns:

```text
runtime guards
      ↓
graph signature/cache key
      ↓
cache hit / miss
      ↓
re-capture / recompile accounting
```

Do not implement those concerns on this frozen branch.
