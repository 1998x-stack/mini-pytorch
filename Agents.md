# Agents.md — mini-pytorch Development Contract

> **Mandatory:** every future iteration reads this file before changing code, tests, docs, branches, version scope or architecture.
>
> This file is canonical for `main`. Frozen version branches carry branch-specific historical contracts.

## 1. Mission

mini-pytorch is a NumPy-first educational framework for learning real PyTorch architectural boundaries.

Priority order:

```text
architectural clarity
      ↓
correctness / invariants
      ↓
inspectability
      ↓
reference proofs / tests
      ↓
measurement
      ↓
optimization
```

Do not optimize for API completeness. Do not copy upstream PyTorch source.

## 2. Branch workflow

```text
validated main
   ↓
create version branch
   ↓
branch version.md + Agents.md
   ↓
implement one dominant learning goal
   ↓
full tests + docs + example
   ↓
exact release-candidate HEAD CI green
   ↓
freeze branch
   ↓
exact frozen HEAD CI green
   ↓
fast-forward into main
   ↓
restore canonical README / Agents / Plan
   ↓
exact main HEAD CI green
```

`main` must remain the newest integrated validated state.

## 3. Completed 0.x sequence

```text
v0.1  Tensor / Dynamic Autograd                           ✅
v0.2  Optimizer state / LR Scheduler                     ✅
v0.3  Dataset / Sampler / DataLoader / MNIST             ✅
v0.4  Storage / stride / views / version safety          ✅
v0.5  Operator Registry / Dispatcher / Redispatch        ✅
v0.6  Neural-network operator stack / CNN                ✅
v0.7  Device / Backend abstraction                       ✅
v0.8  Advanced Autograd Engine                           ✅
v0.9  Training Runtime / Serialization / Reproducibility ✅
v0.10 Distributed / DDP Concepts                         ✅
v0.11 Graph IR / Eager-to-Graph Capture                  ✅
v0.12 Guards / Graph Cache / Recompile                   ✅
v0.13 AOTAutograd-like Graph Differentiation             ✅
v0.14 Lowering / Fusion / Inductor-like Concepts         ✅
v0.15 Runtime / Performance Mechanics                    ✅
v1.0  Integrated Architecture                            ← next
```

Current integrated validation baseline:

```text
v0.15
247 tests
OK
```

## 4. Frozen architecture invariants through v0.15

### Tensor / memory

- Storage owns backend allocation and shared version counter;
- shape/stride/storage_offset define Tensor layout;
- views may alias Storage;
- `view`, `reshape`, and `contiguous` have different allocation semantics;
- in-place changes participate in version safety.

### Dispatcher / backend

- logical operators route through schema/registry/Dispatcher;
- Autograd/NoGrad are policy keys;
- CPU/EDU are backend keys;
- redispatch removes handled keys rather than recursively re-entering the same policy;
- mixed-device input fails explicitly.

### Autograd

- eager backward is owned by `AutogradEngine`;
- graph lifetime and saved-value version safety are distinct;
- retain_graph / retain_grad / hooks remain separate concepts;
- custom Function uses the same Engine lifecycle;
- no fake higher-order-gradient claim.

### Training / distributed

- checkpoint includes model/optimizer/scheduler/RNG/progress state;
- deterministic resume is tested;
- DDP simulation separates local gradient readiness from explicit synchronization;
- optimizer remains unaware of DDP communication.

### Compiler

- capture records outer logical Dispatcher operations, not redispatch internals;
- capture keeps observed Tensor objects alive so Python id reuse cannot merge Graph Values;
- Graph IR does not bake in backend/Autograd decisions;
- guard failure is an inspectable specialization miss;
- GraphBreak never installs partial cache state;
- AOT backward uses explicit saved Values and gradient inputs;
- source Graphs are not mutated by lowering passes;
- pointwise fusion remains conservative;
- ExternalOps still execute through Dispatcher.

### Runtime planning

- benchmark/profiling results are measurement, not correctness assumptions;
- Graph output and AOT retained Values escape normal temporary lifetime;
- buffer reuse requires explicit compatible non-overlapping lifetimes;
- only safe pointwise-fusion temporaries use planner-owned output buffers in v0.15;
- `LoweredProgram` describes work; `ExecutionPlan` describes runtime resource assignment;
- `GraphInterpreter == LoweredExecutor == PlannedExecutor` is a required correctness chain.

## 5. Cross-layer rules

- separate policy from mechanism;
- avoid circular core imports;
- numerical kernels must not depend on Module/Optimizer/DataLoader/compiler policy;
- Autograd policy stays outside backend numerical kernels;
- compiler/runtime layers do not create a second Tensor architecture;
- unsupported behavior fails explicitly;
- historical reference executors remain correctness oracles for newer layers;
- performance claims require repeated measurement.

## 6. Testing contract

Every meaningful change should include, where applicable:

- happy path;
- invariants/property tests;
- error path;
- numerical/reference validation;
- integration proof;
- full historical regression.

Run:

```bash
python -m unittest discover -s tests -v
```

Never claim a test count or validated state unless that exact code state ran successfully.

## 7. Documentation contract

Review on every version/release:

```text
README.md
version.md
Agents.md
MANIFEST.txt
Plan.md on main
relevant docs/*
relevant example
```

## 8. Frozen branch rule

Frozen branches may receive only:

- documentation corrections;
- tests clarifying already-promised behavior;
- narrow bug fixes inside that historical version boundary.

Future development begins from latest `main`.

## 9. v1.0 directive — integrate, do not expand

The 0.x sequence already contains the major planned mechanisms. v1.0 should not introduce another large subsystem merely to increase scope.

Central question:

> Can the complete mini-pytorch stack now be explained, exercised and validated as one coherent architecture?

Required v1.0 work:

```text
architecture audit
      ↓
API / naming consistency audit
      ↓
cross-version documentation reconciliation
      ↓
one end-to-end integrated example
      ↓
mini-pytorch ↔ real PyTorch subsystem map
      ↓
full regression / release snapshot
```

### v1.0 allowed work

- small API cleanup where it improves consistency without hiding learning boundaries;
- missing integration tests across existing subsystems;
- documentation/diagram improvements;
- one coherent example that connects eager, training, compiler and runtime layers;
- subsystem mapping to real PyTorch concepts;
- release metadata and final validation.

### v1.0 non-goals

- new accelerator backend;
- Triton/CUDA codegen;
- symbolic shape engine;
- production allocator;
- production DDP/NCCL;
- new compiler family;
- API feature-parity sprint.

## 10. v1.0 Definition of Done

v1.0 freezes only when:

1. architecture docs match actual source boundaries;
2. public learning APIs/names are internally consistent;
3. an end-to-end example crosses the major layers without hidden shortcuts;
4. PyTorch subsystem mapping is explicit and caveated;
5. historical version branches remain understandable;
6. all tests are green at exact release HEAD;
7. README/version/Agents/Plan/MANIFEST are release-ready;
8. no unsupported production/performance claim is made.

## 11. Current directive

`main` contains completed **v0.15 Runtime / Performance Mechanics**.

Next milestone: **v1.0 — Integrated Architecture**.
