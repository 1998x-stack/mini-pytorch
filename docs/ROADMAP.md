# mini-pytorch Roadmap — Completed Learning Sequence

> v1.0 replaces the old forward-looking roadmap with the actual completed version history.
>
> The original project goal was to remove one layer of "PyTorch magic" at a time with readable Python/NumPy mechanisms. That 0.x mechanism-building sequence is now complete.

## 1. Version history

| Version | Theme | Dominant learning boundary | Validation milestone |
|---|---|---|---:|
| `v0.1` | Dynamic Autograd MVP | Tensor → Function/GradFn → backward | historical |
| `v0.2` | Optimizers / LR schedulers | optimizer state, parameter groups, scheduler state | historical |
| `v0.3` | Data pipeline | Dataset → Sampler → BatchSampler → DataLoader | historical |
| `v0.4` | Tensor memory model | Storage / stride / views / version safety | historical |
| `v0.5` | Dispatcher | schemas / DispatchKeys / redispatch | 80 |
| `v0.6` | CNN operators | Conv / Pool / BatchNorm / Dropout | 90 |
| `v0.7` | Device / Backend | logical placement + backend allocation/dispatch | 103 |
| `v0.8` | Advanced Autograd Engine | graph lifetime / hooks / retain-grad | 123 |
| `v0.9` | Training Runtime | RNG / checkpoint / deterministic resume | 140 |
| `v0.10` | Distributed / DDP | ProcessGroup / reducer / buckets / sharding | 159 |
| `v0.11` | Graph IR | Dispatcher-level eager capture / replay | 176 |
| `v0.12` | Guards / Cache | specialization / guard failure / recompile | 192 |
| `v0.13` | AOT-style differentiation | explicit forward/backward Graph ABI | 206 |
| `v0.14` | Lowering / Fusion | analysis / transforms / LoweredProgram | 221 |
| `v0.15` | Runtime / Performance | profiling / lifetimes / ExecutionPlan | 247 |
| `v1.0` | Integrated Architecture | audit / cross-layer proof / final learning release | release target |

The numeric test counts above are historical validated branch milestones where recorded. v1.0 gets its own final count only after exact release-head CI passes.

---

## 2. Completed architecture chain

```text
Tensor / Dynamic Autograd
      ↓
Module / Optimizer
      ↓
Dataset / Sampler / DataLoader
      ↓
Storage / stride / views
      ↓
Dispatcher / Backend
      ↓
Advanced Autograd Engine
      ↓
Training Runtime / Checkpoint / RNG
      ↓
Distributed / DDP
      ↓
Graph IR / Capture
      ↓
Guards / Graph Cache
      ↓
AOT forward/backward Graphs
      ↓
Analysis / Lowering / Fusion
      ↓
Runtime profiling / lifetimes
      ↓
MemoryPlan / ExecutionPlan
      ↓
PlannedExecutor
```

v1.0 exists to make this chain coherent as one release rather than add another box below it.

---

## 3. Historical branch policy

Every historical version branch is a learning snapshot.

When reading one:

```text
Agents.md
   ↓
version.md
   ↓
README.md / version-specific docs
   ↓
source + tests
```

Frozen branches are not ordinary development bases.

Allowed historical corrections remain limited to:

- documentation corrections;
- tests clarifying already-promised behavior;
- narrow bug fixes inside that historical boundary.

New development always starts from the newest validated `main`.

---

## 4. v1.0 completion plan

v1.0 is a consolidation release with seven workstreams:

```text
1. architecture audit
2. public API / naming audit
3. cross-layer integration tests
4. integrated example
5. PyTorch conceptual mapping
6. canonical documentation reconciliation
7. release-quality validation
```

Required artifacts:

```text
docs/ARCHITECTURE_V1.md
docs/API_AUDIT.md
docs/PYTORCH_MAPPING.md
examples/19_integrated_architecture.py
tests/test_v1_integration.py
```

and updated:

```text
README.md
docs/ARCHITECTURE.md
docs/LEARNING_PATH.md
docs/ROADMAP.md
MANIFEST.txt
package version metadata
```

---

## 5. Explicitly closed scope for the original roadmap

The following are interesting future research directions but are **not missing v1.0 requirements**:

```text
real CUDA backend
Triton/code generation
symbolic dynamic shapes
multiprocessing/prefetch DataLoader
pinned-memory/stream experiments
real NCCL/network ProcessGroup
production caching allocator
whole-Python/bytecode compiler
full higher-order autograd
PyTorch API parity
```

The original learning sequence is complete without these because the architectural ownership boundaries they would extend are already explicit.

---

## 6. After v1.0

There is no automatic `v1.1` feature queue.

Any post-v1.0 work should begin with a new question such as:

```text
What new mental model is missing?
Why can it not be learned from the v1.0 architecture?
Which existing boundary should own it?
What is the smallest readable implementation that teaches it?
```

Only then should a new roadmap be created.

This prevents the project from drifting from:

```text
architecture learning framework
```

into:

```text
unbounded clone-by-feature-count
```

---

## 7. Final learning objective

The project succeeds when a reader can take:

```python
loss = ((model(x) - target) ** 2).mean()
loss.backward()
optimizer.step()
```

and expand the hidden mechanisms into a coherent model of:

```text
Tensor layout
Dispatcher/backend execution
eager autograd
module/parameter ownership
optimizer state
data ordering/batching
runtime reproducibility
DDP synchronization
Graph capture and guards
AOT backward
lowering/fusion
runtime lifetime and buffer planning
```

v1.0 is the release that documents and validates that complete explanation.
