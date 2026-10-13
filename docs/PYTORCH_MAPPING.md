# mini-pytorch ↔ PyTorch Concept Mapping

> v1.0 conceptual map.
>
> This document explains which real PyTorch subsystem family each mini-pytorch mechanism is intended to illuminate. It does **not** claim source equivalence, API parity, performance parity, or identical edge-case semantics.

## 1. Mapping table

| mini-pytorch concept | Real PyTorch concept family | What the analogy teaches | Important simplification |
|---|---|---|---|
| `Device` | `torch.device` / backend identity | placement is part of Tensor execution semantics | only `cpu` + host-backed `edu`; no real accelerator runtime |
| `Backend` | backend allocation/kernel families | allocation/copy mechanism belongs below Tensor policy | NumPy-backed only; tiny interface |
| `Storage` | Tensor Storage + TensorImpl layout concepts | Tensor metadata can view shared underlying storage | Python/NumPy model; no C++ TensorImpl object |
| shape/stride/offset views | strided Tensor/view semantics | transpose/view can share storage and become non-contiguous | subset of indexing/layout semantics |
| shared `VersionCounter` | Autograd in-place version tracking | saved backward state must detect invalidating mutation | one educational counter per Storage |
| `OperatorSchema` / `Library` | operator schema/registration concepts | logical operator definition is separate from backend kernel | tiny Python registry; no codegen/native_functions pipeline |
| `Dispatcher` / `DispatchKeySet` | PyTorch dispatcher / dispatch keys | policy and backend implementations compose by dispatch priority | fixed small key set; no full TLS/boxed dispatcher machinery |
| `Autograd` / `NoGrad` dispatch keys | autograd-related dispatcher policy | Autograd can wrap/redispatch to backend execution | educational fallback/kernel model |
| CPU / EDU keys | backend dispatch keys | same logical operator may select different backend implementations | EDU is simulated on NumPy memory |
| `GradFn` / `Context` | autograd graph Node / saved tensors | eager forward records reverse-mode dependencies dynamically | Python objects and explicit formulas |
| `AutogradEngine` | PyTorch autograd engine | backward is graph execution, gradient accumulation and graph-lifetime management | single-threaded; no higher-order graph construction |
| `Module` / `Parameter` / buffers | `torch.nn.Module` / `Parameter` / buffers | parameter ownership is separate from differentiation | much smaller hook/state surface |
| `Optimizer` / parameter groups | `torch.optim` | optimizers consume gradients; they do not compute them | reduced algorithm/option set |
| LR schedulers | `torch.optim.lr_scheduler` | LR policy is state layered above optimizer groups | reduced parity and serialization behavior |
| `Dataset` | `torch.utils.data.Dataset` | sample access is a separate responsibility | map-style educational subset |
| `Sampler` / `BatchSampler` | `torch.utils.data.Sampler` / batching order | index order and batching are policies, not Dataset behavior | limited sampler family |
| `DataLoader` | `torch.utils.data.DataLoader` | Dataset + sampling + collation compose into an iterable | single-process; no workers/prefetch/pinned memory |
| framework RNG | RNG state / generators concept | deterministic training needs explicit random-state ownership | one framework NumPy-backed stream |
| `CheckpointState` | model/optimizer state_dict + serialization workflows | reproducibility requires more than model weights | custom JSON + NPZ directory format |
| `ProcessGroup` | `torch.distributed` / c10d ProcessGroup family | collective communication is a separate runtime abstraction | deterministic single-process simulation; no transport |
| `Reducer` / `GradientBucket` | DDP reducer/bucketing concepts | gradient readiness can be grouped before synchronization | no real async overlap/network |
| `DistributedDataParallel` | `torch.nn.parallel.DistributedDataParallel` | model wrapper coordinates gradient synchronization, optimizer remains ordinary | simulated ranks in one process |
| `DistributedSampler` | `torch.utils.data.distributed.DistributedSampler` | data sharding belongs at sampling layer | educational deterministic partitioning |
| `Graph` / `Node` / `Value` | graph IR / FX-like dataflow concepts | eager logical ops can be represented explicitly | custom tiny IR; not FX |
| Dispatcher-level `capture()` | graph capture/tracing concept | one logical eager operation can become one IR node | observes supported Dispatcher ops, not Python bytecode |
| `InputSignature` / guards | TorchDynamo/`torch.compile` specialization guards concept | compiled artifacts are valid only under assumptions | only Tensor signature/grad-mode guards |
| `GraphCache` / `CompiledFunction` | `torch.compile` guarded specialization/cache concept | guard failure can select/recompile a different graph | in-memory educational cache, no frame/code-object compiler |
| `AOTProgram` / `differentiate_graph()` | AOTDispatcher/AOTAutograd concept family | backward can be made an explicit graph with a saved-value ABI | small hand-written VJP registry |
| Graph analysis / DCE / constant folding | compiler graph passes | transformations should consume explicit IR invariants | tiny pure-graph assumption |
| `FusionGroup` / `PointwiseFusion` | Inductor-like fusion/lowering concepts | lowering can group compatible work without changing user Tensor semantics | add/mul/relu reference fusion only; no Triton/codegen |
| `LoweredProgram` | lowered compiler representation | compiler output can be lower-level than the source Graph | Python instruction objects |
| benchmark/profiler | `torch.utils.benchmark` / profiler concept family | performance evidence is separate from correctness | simple wall-clock per-instruction timing |
| `ValueLifetimeAnalysis` | compiler/runtime liveness analysis | buffer reuse requires last-use knowledge | static concrete shapes only |
| `MemoryPlan` / `BufferSlot` | compiler/runtime memory planning concepts | logical Values may reuse fewer physical allocations | conservative exact-shape/dtype/device greedy planner |
| `ExecutionPlan` | compiler-runtime handoff concept | operation order and runtime resource assignment are different abstractions | project-specific educational representation, not a public PyTorch class |
| `PlannedExecutor` | generated/compiled runtime execution concept | a validated plan can execute with explicit temporary buffers | Python/NumPy executor; ExternalOps still use mini Dispatcher |

---

## 2. Tensor, Storage and views

PyTorch's public Tensor documentation describes a Tensor as a multidimensional strided view over associated storage, and the Tensor Views documentation emphasizes that views share underlying data.

mini-pytorch makes this visible in Python:

```text
Storage
  ↑
Tensor(shape, stride, storage_offset)
```

The educational goal is the same mental model:

```text
Tensor identity != allocation identity
```

What mini-pytorch does **not** model is the full C++ `TensorImpl`/Storage implementation, all layouts, sparse tensors, quantization, meta tensors, or allocator behavior.

---

## 3. Dispatcher

PyTorch's dispatcher is substantially richer than the mini implementation. Current PyTorch documentation/devlogs describe per-tensor dispatch keys, global dispatch tables and context/thread-local included/excluded key behavior.

mini-pytorch keeps only the learning core:

```text
operator schema
     +
DispatchKeySet
     ↓
priority selection
     ↓
kernel/fallback
     ↓
redispatch
```

This is enough to understand why these concerns should remain separate:

```text
logical op
backend implementation
autograd policy
no-grad policy
compiler observation
```

It is not enough to reproduce PyTorch's complete key set, TLS behavior, autocast, functionalization, Python dispatch modes or operator code generation.

---

## 4. Eager Autograd

PyTorch's Autograd documentation describes reverse-mode differentiation as a graph rebuilt dynamically from executed operations, with `grad_fn`/graph Nodes and saved tensors supporting backward.

mini-pytorch exposes the same learning ideas directly:

```text
Tensor
  ↓
GradFn(function, parents, Context)
  ↓
AutogradEngine
```

Version counters in mini-pytorch deliberately teach the same class of in-place correctness problem documented by PyTorch: mutation can invalidate values saved for backward.

Differences are large: mini-pytorch is single-threaded, uses NumPy gradient arrays internally, has a small Function set, and explicitly does not support `create_graph=True`.

---

## 5. Module / Optimizer / Data

These boundaries intentionally resemble the high-level PyTorch organization:

```text
Module organizes parameters
Autograd computes gradients
Optimizer consumes gradients
Dataset exposes samples
Sampler chooses indices
DataLoader fetches/collates batches
```

The important point is architectural responsibility, not exact API coverage.

Real PyTorch DataLoader supports iterable datasets, multiprocessing, worker lifecycle, memory pinning, prefetch and many configuration options. mini-pytorch v1.0 intentionally stops at the single-process map-style learning boundary.

---

## 6. Distributed concepts

Real PyTorch distributed training uses ProcessGroup backends and supports collectives across processes/devices; DDP coordinates gradient synchronization around normal model/optimizer usage.

mini-pytorch models:

```text
rank/world_size
collective rendezvous semantics
Reducer
GradientBucket
DDP wrapper
DistributedSampler
```

but runs deterministically in one process. There is no NCCL/Gloo transport, `torchrun`, process spawning, network failure model, real asynchronous overlap or GPU stream interaction.

Therefore the mapping should be read as:

```text
same responsibility boundary
!=
same runtime implementation
```

---

## 7. Graph capture and `torch.compile` concepts

Current `torch.compile` documentation describes guarded compilation, cached compiled results, graph breaks and recompilation when previous compiled results no longer apply. PyTorch's implementation works at a much deeper Python/compiler level than mini-pytorch.

mini-pytorch teaches only the shape of the mechanism:

```text
runtime Tensor signature
      ↓
GuardSet
      ↓
GraphCache
   ├── hit → reuse Graph
   └── miss → capture specialization
```

The capture mechanism is intentionally different:

```text
mini-pytorch:
observe outer Dispatcher-visible logical ops

PyTorch compiler family:
works with Python/frame/graph compiler machinery far beyond this project
```

Do not interpret `CompiledFunction` as a reimplementation of TorchDynamo.

---

## 8. AOT backward

Current PyTorch compiler documentation refers to the component producing a compiled backward graph as AOTDispatcher, historically/commonly known as AOTAutograd.

mini-pytorch's v0.13/v1.0 path mirrors the learning contract:

```text
forward Graph
      ↓
explicit saved-value requirements
      ↓
backward Graph
```

The crucial lesson is the ABI between forward and backward:

```text
forward output
+
saved Values
+
grad_output
      ↓
backward Graph
      ↓
input gradients
```

mini-pytorch uses a small explicit VJP registry instead of PyTorch's production operator/autograd/compiler machinery.

---

## 9. Lowering, fusion and Inductor-like ideas

PyTorch tutorials describe a compiler stack where Dynamo capture can feed AOT Autograd/AOTDispatcher and Inductor, with Inductor lowering/fusing operations and generating optimized implementations.

mini-pytorch stops at a readable reference model:

```text
Graph
 ↓
analysis / DCE / constant folding
 ↓
use-aware pointwise fusion
 ↓
LoweredProgram
 ↓
reference executor
```

There is no Triton kernel generation, C++ code generation, template/autotuning system, dynamic shape compiler or production scheduler.

The project's `PointwiseFusion` should therefore be read as:

> a small executable demonstration of why a lower-level compiler representation exists,

not as an implementation of TorchInductor.

---

## 10. ExecutionPlan has no claimed one-to-one PyTorch public class

`ExecutionPlan`, `MemoryPlan` and `BufferSlot` are project-specific educational abstractions.

They teach general compiler/runtime ideas:

```text
Value liveness
buffer compatibility
physical allocation reuse
release schedule
compiler/runtime handoff
```

Real PyTorch/Inductor/runtime internals contain much richer scheduling and memory-planning mechanisms, but mini-pytorch intentionally does not claim a stable one-to-one class correspondence.

This caveat is important: a good teaching abstraction can represent a real architectural problem without pretending to match upstream source organization exactly.

---

## 11. Recommended comparison workflow

When using mini-pytorch to understand PyTorch internals, compare responsibilities in this order:

```text
mini-pytorch source
      ↓
What decision does this object own?
      ↓
PyTorch public/dev documentation
      ↓
Which subsystem family owns the analogous decision?
      ↓
List the simplifications explicitly
```

Avoid the weaker approach:

```text
same class name
⇒
must be the same implementation
```

---

## 12. Official PyTorch references used for the v1.0 audit

The mapping above was checked against current PyTorch documentation available during the v1.0 audit:

- Tensor / Storage / strided Tensor model: https://docs.pytorch.org/docs/main/tensors.html
- Tensor views: https://docs.pytorch.org/docs/main/tensor_view.html
- Dispatcher devlog (2026): https://docs.pytorch.org/devlogs/dispatcher/2026-04-16-how-does-the-dispatcher-work/
- Dispatcher extension tutorial: https://docs.pytorch.org/tutorials/advanced/dispatcher.html
- Autograd mechanics: https://docs.pytorch.org/docs/stable/notes/autograd
- `torch.utils.data`: https://docs.pytorch.org/docs/2.14/data.html
- Distributed package: https://docs.pytorch.org/docs/stable/distributed.html
- DDP tutorial: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
- `torch.compile`: https://docs.pytorch.org/docs/stable/generated/torch.compile
- Compiler backward/AOTDispatcher semantics: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html
- Inductor debugging/tutorial stack description: https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html

These references can evolve. mini-pytorch's historical branches remain fixed learning snapshots even when upstream PyTorch changes terminology or implementation details.
