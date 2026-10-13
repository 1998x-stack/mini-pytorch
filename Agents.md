# Agents.md — mini-pytorch v1.0 Development Contract

> Read this file before changing branch `v1.0`.

## 1. Branch identity

- Branch: `v1.0`
- Previous: `v0.15`
- Theme: Integrated Architecture
- Status: active until exact frozen-head CI passes
- Starting baseline: **247 tests / OK**
- Base main commit: `56191655f1188134e62e965dc58fdd3dcd3d062b`

## 2. Dominant learning goal

v1.0 is a consolidation release.

The question is not “what new subsystem should be added?” but:

> Can a reader trace one Tensor expression through the complete architecture, understand who owns each decision, and verify the cross-layer behavior with executable proofs?

## 3. Required reading order

Before modifying v1.0:

1. `Agents.md`
2. `version.md`
3. `README.md`
4. relevant current source
5. relevant historical mechanism docs
6. tests for the touched boundaries

If a source boundary is unclear, inspect the implementation before changing documentation. Documentation must describe the code, not an imagined architecture.

## 4. Allowed v1.0 work

Allowed:

- architecture audit/documentation;
- small compatibility-preserving API/naming cleanup;
- focused cross-layer integration tests;
- integrated educational example;
- PyTorch subsystem concept mapping;
- canonical documentation reconciliation;
- package/release metadata updates;
- narrow fixes exposed by integration work when they remain inside existing subsystem contracts.

## 5. Forbidden scope expansion

Do not introduce:

- CUDA/Triton codegen;
- another backend family;
- symbolic shape engine;
- multiprocessing/prefetch DataLoader architecture;
- NCCL/network ProcessGroup;
- production caching allocator;
- bytecode/Python compiler;
- a new compiler/runtime subsystem;
- feature-parity work justified only by “PyTorch has it”.

If integration exposes a missing major mechanism, document it as a future limitation rather than silently expanding v1.0.

## 6. Architecture ownership rules

Preserve these existing boundaries.

### Tensor / memory

```text
Storage owns allocation + shared version counter
Tensor owns shape / stride / offset / requires_grad metadata
Device describes logical placement
```

Tensor memory semantics must not move into compiler/runtime planning.

### Dispatcher / backend

```text
logical operator
   ↓
Dispatcher policy
   ↓
backend kernel
```

Backend kernels must not depend on Module, Optimizer, DataLoader, compiler cache or DDP policy.

### Autograd

`AutogradEngine` owns eager backward graph execution. Saved-value version checks and graph lifetime are separate concerns. Do not create a second autograd runtime inside compiler integration tests.

### Training/runtime

Module/Optimizer/Data compose above Tensor/Dispatcher. Checkpoint/RNG/progress state is runtime state, not a Tensor-kernel concern.

### Distributed

DDP/Reducer/ProcessGroup simulate synchronization outside optimizer algorithms and numerical kernels.

### Compiler

```text
capture
  ↓
Graph IR
  ↓
guards/cache
  ↓
AOT differentiation
  ↓
analysis/transforms/lowering
  ↓
LoweredProgram
```

Compiler passes must not mutate source Graphs unless the API explicitly promises mutation.

### Planned runtime

```text
LoweredProgram = what executes
ExecutionPlan = runtime resource assignment
```

ExternalOps remain Dispatcher-owned. Planner-owned output buffers remain limited to explicitly safe pointwise temporaries.

## 7. Public API audit rules

When auditing exports:

- prefer stable learning concepts over implementation helpers;
- do not remove a public name without checking repository docs/examples/tests;
- do not rename merely to mimic PyTorch spelling;
- internal/private helpers should not become public just to make the integrated example shorter;
- integration examples should import from supported public module paths where such paths already exist.

Any API cleanup must have a compatibility rationale and tests where behavior changes.

## 8. Integrated example rules

`examples/19_integrated_architecture.py` must separate phases explicitly:

```text
training-time
runtime/checkpoint state
compiler-time
planned runtime
verification
```

Do not imply that graph compilation/checkpoint/DDP are all necessarily executed inside one production loop.

The example should be deterministic, self-contained and require no external dataset/network download.

## 9. PyTorch mapping rules

`docs/PYTORCH_MAPPING.md` is conceptual correspondence only.

Use phrasing such as:

```text
concept family
educational analogue
important simplification
```

Do not claim source equivalence, API parity, performance parity or production behavior.

## 10. Testing contract

Required v1.0 integration proofs should cover only meaningful cross-layer seams:

```text
Tensor / Dispatcher / Backend / Autograd
Module / Optimizer / Runtime checkpoint
Capture / Cache / AOT / Lowering / ExecutionPlan
```

Do not create one huge brittle test that duplicates the entire historical suite.

Full regression command:

```bash
python -m unittest discover -s tests -v
```

Never claim a test count until the exact code state ran successfully.

## 11. Documentation contract

Review before release:

```text
README.md
version.md
Agents.md
MANIFEST.txt
docs/ARCHITECTURE.md
docs/ARCHITECTURE_V1.md
docs/LEARNING_PATH.md
docs/ROADMAP.md
docs/PYTORCH_MAPPING.md
examples/19_integrated_architecture.py
```

`Plan.md` remains canonical on `main`; the v1.0 branch uses `version.md` + this `Agents.md` for release scope.

Historical docs may preserve historical “next version” statements if clearly version-specific. Canonical current docs must not be stale.

## 12. Release workflow

```text
active v1.0
   ↓
architecture/docs/tests/example complete
   ↓
package version 1.0.0
   ↓
full regression
   ↓
exact release-candidate HEAD CI green
   ↓
freeze version.md + Agents.md
   ↓
exact frozen HEAD CI green
   ↓
fast-forward v1.0 into main
   ↓
restore canonical integrated README / Agents / Plan
   ↓
exact main HEAD CI green
```

No future work begins from a stale pre-integration branch.

## 13. Definition of Done

v1.0 is complete only when:

- architecture ownership is documented from actual code;
- API/naming audit is complete;
- integrated example runs without external downloads;
- cross-layer tests pass;
- PyTorch mapping is explicit and caveated;
- docs agree on terminology and baseline;
- package reports `1.0.0`;
- release-candidate exact HEAD CI is green;
- frozen exact HEAD CI is green;
- validated tree is integrated into `main`;
- final main exact HEAD CI is green.
