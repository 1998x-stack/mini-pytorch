# Architecture

> Current architecture index for mini-pytorch v1.0.
>
> For the full ownership/dependency audit, read [`ARCHITECTURE_V1.md`](ARCHITECTURE_V1.md). Historical subsystem details remain in the version-specific documents in this directory.

## 1. Complete stack

```text
Python API
   ↓
Tensor / Storage / Device
   ↓
Dispatcher / Backend
   ↓
Eager AutogradEngine
   ↓
Module / Optimizer / Data
   ↓
Training Runtime / Checkpoint / RNG
   ↓
Distributed / DDP simulation

Python Tensor function
   ↓
Graph capture
   ↓
Signatures / Guards / GraphCache
   ↓
AOT forward + backward Graphs
   ↓
Analysis / Lowering / Fusion
   ↓
LoweredProgram
   ↓
Profiling / Lifetime Analysis
   ↓
MemoryPlan / ExecutionPlan
   ↓
PlannedExecutor
```

## 2. Five boundaries to remember

### Tensor memory

```text
Storage = allocation + shared version
Tensor = shape + stride + offset + autograd-facing metadata
```

### Operator execution

```text
logical operator
      ↓
Dispatcher
      ↓
backend kernel
```

Autograd/NoGrad policy is kept above backend numerical kernels.

### Training ownership

```text
Module organizes parameters
Autograd computes gradients
Optimizer updates parameters
DataLoader supplies batches
Runtime preserves reproducible state
DDP synchronizes gradients
```

These are separate responsibilities.

### Compiler ownership

```text
Graph = logical dataflow
LoweredProgram = lowered work
ExecutionPlan = runtime resource assignment
```

### Correctness references

For supported compiled execution:

```text
Eager
 == GraphInterpreter
 == LoweredExecutor
 == PlannedExecutor
```

For supported AOT gradients, eager Autograd remains the numerical reference.

## 3. Recommended architecture reading order

```text
TENSOR_MEMORY.md
DEVICE_BACKEND.md
DISPATCHER.md
AUTOGRAD_ENGINE.md
TRAINING_RUNTIME.md
DISTRIBUTED_DDP.md
GRAPH_IR.md
GRAPH_CACHE.md
AOT_AUTOGRAD.md
LOWERING_FUSION.md
RUNTIME_PERFORMANCE.md
ARCHITECTURE_V1.md
PYTORCH_MAPPING.md
```

## 4. v1.0 scope

v1.0 integrates the existing mechanism chain. It does not claim:

- PyTorch API parity;
- production CUDA/Triton code generation;
- production multiprocessing DataLoader;
- real network/NCCL DDP;
- symbolic dynamic-shape compilation;
- production allocator/profiler behavior;
- performance parity.

The architecture is considered complete for the project's original learning sequence because every major ownership boundary is now executable and testable in a readable Python/NumPy implementation.
