# Agents.md — mini-pytorch v0.11 Development Contract

> Read this file before modifying branch `v0.11`.

## Branch identity

- Branch: `v0.11`
- Previous version: `v0.10`
- Primary mental model: **Graph IR / Eager-to-Graph Capture**
- Status: active development
- Target package version: `0.11.0`

## 1. Scope

Build one compiler-facing layer only:

```text
logical eager operator
      ↓
Dispatcher
      ├── normal execution
      └── capture observer
              ↓
       Value / Node / Graph
              ↓
       GraphInterpreter
              ↓
          Dispatcher replay
```

Capture must reuse the existing logical operator boundary. Do not create duplicate numerical kernels or duplicate Autograd formulas.

## 2. Mandatory architecture rules

- Dispatcher remains the owner of operator routing.
- Capture observes only outer `dispatch()` calls; internal `redispatch()` must not create duplicate nodes.
- Graph replay must call Dispatcher again.
- IR must be deterministic and inspectable.
- Graph inputs, Tensor constants and Python literals must be distinguishable.
- Unsupported dataflow must raise an explicit GraphBreak rather than silently baking dynamic values as constants.
- Eager behavior outside capture must remain unchanged.
- Capturing must not disable Autograd; the trace run is a normal eager run plus observation.
- Replay outputs must be ordinary Tensors and must support ordinary backward.

## 3. IR boundary

Required core objects:

```text
Value
Node
Graph
```

Value kinds should cover at least:

```text
input
constant
intermediate
```

Node records logical operator name, structured positional/keyword arguments and structured outputs.

Do not encode Python object identity as the persistent graph format.

## 4. Capture boundary

Provide a context/API that:

1. registers explicit example Tensor inputs as Graph inputs;
2. executes the Python function eagerly;
3. observes supported Dispatcher calls;
4. records produced Tensors as intermediate Values;
5. marks returned Tensor values as Graph outputs;
6. returns both the Graph and eager output for validation/use.

Nested active capture is rejected in v0.11.

## 5. GraphBreak rule

A Tensor flowing into a captured logical operator must be one of:

- a registered graph input;
- a captured intermediate;
- an explicitly materialized Tensor constant allowed by the capture boundary.

If an untracked Tensor was dynamically produced during the capture by an unsupported eager operation, do not silently freeze it as a constant. Raise `GraphBreakError` with the operator/dataflow context.

v0.11 may conservatively restrict constants instead of pretending to support arbitrary Python/Module capture.

## 6. Interpreter rule

GraphInterpreter executes nodes in order and resolves IR Values to runtime objects.

It must:

- bind runtime graph inputs;
- materialize graph constants;
- recursively resolve structured argument specs;
- call the normal Dispatcher;
- bind node outputs;
- return Graph outputs in their recorded structure.

It must not call backend kernels directly.

## 7. Correctness contract

At minimum tests must cover:

- deterministic IR IDs/text;
- graph input/constants/intermediates;
- add/mul/matmul/relu/sum capture;
- kwargs/literal argument capture;
- replay forward equality;
- replay backward/input-gradient equality;
- CPU and EDU replay;
- nested capture rejection;
- wrong input count;
- GraphBreak from unsupported eager dataflow;
- replay through Dispatcher, not direct kernels;
- all historical 159 tests remain green.

## 8. Explicit non-goals

Do not implement in v0.11:

- guards/cache (`v0.12`);
- Python bytecode/frame tracing;
- symbolic control flow;
- full Module/Parameter lifting;
- AOTAutograd (`v0.13`);
- optimization/fusion/lowering (`v0.14`);
- Triton/code generation;
- performance claims.

## 9. Freeze contract

Before freezing:

```text
implementation
+ tests
+ docs/GRAPH_IR.md
+ examples/14_graph_capture.py
+ README.md
+ MANIFEST.txt
+ version.md
+ exact final branch-head CI
```

After exact final-head CI is green, mark this branch frozen, integrate it to `main`, restore canonical main docs, and advance `Plan.md` to v0.12 Guards / Graph Cache.
