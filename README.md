# mini-pytorch v0.15 — Runtime / Performance Mechanics

> 在 v0.14 已经得到正确、可检查的 `LoweredProgram` 之后，v0.15 继续回答：runtime 如何**测量**执行、分析 Value 生命周期、规划 buffer、执行安全复用，并用可重复 benchmark 判断优化是否真的有收益？

**Branch:** `v0.15`  
**Package:** `0.15.0`  
**Previous:** `v0.14` Lowering / Fusion  
**Validation baseline:** `247 / 247 tests passed` (release-candidate implementation)

## 1. 本版本核心差异

v0.14：

```text
Graph
  ↓
analysis / transforms / fusion
  ↓
LoweredProgram
  ↓
LoweredExecutor
```

v0.15：

```text
LoweredProgram
      ↓
benchmark + profiling
      ↓
Value lifetime analysis
      ↓
Buffer interval analysis
      ↓
MemoryPlan
      ↓
ExecutionPlan
      ↓
PlannedExecutor
```

关键思想：

```text
LoweredProgram
= what executes

ExecutionPlan
= how runtime resources are assigned
```

## 2. Measurement before optimization

新增：

```python
from mini_pytorch.compiler import BenchmarkConfig, benchmark

report = benchmark(
    fn,
    config=BenchmarkConfig(
        warmup=3,
        repeats=10,
        iterations_per_repeat=5,
    ),
)

print(report.median_ms)
print(report.mean_ms)
```

规则：

- warmup 不进入 samples；
- 每个 repeat 归一化成 per-iteration latency；
- median 为主要摘要；
- 同时暴露 mean/min/max；
- unit tests **不使用绝对速度阈值**；
- benchmark 结果是 evidence，不是 correctness oracle。

## 3. Per-instruction profiling

```python
profiled = profile_lowered(lowered.program, x, y)
print(profiled.profile.format())
```

每个 `ExecutionEvent` 包含：

```text
instruction_index
instruction_kind
label
input Value ids
output Value ids
elapsed_ns
```

`ExternalOp` 与 `PointwiseFusion` 会分别显示。

## 4. Value lifetime

粗粒度 lowered-instruction lifetime：

```text
ValueLifetime
├── birth_instruction
├── last_use_instruction
├── input / constant / output / retained flags
├── shape
├── dtype
└── device
```

约定：

```text
input / constant birth = -1
Graph output last_use = instruction_count
AOT saved/retained Value last_use = instruction_count
```

因此 runtime 不会在 backward 仍需要 forward saved Value 时提前回收它。

## 5. Buffer planning

新增：

```text
BufferSpec
BufferInterval
BufferSlot
MemoryPlan
```

v0.15 的 reuse 很保守。只有同时满足以下条件的 `PointwiseFusion` 临时值才可能进入 reusable slot：

```text
producer 属于 pointwise fusion
不是 graph output
不是 retained/AOT saved value
shape 完全相同
dtype 完全相同
device 完全相同
lifetime 不重叠
```

暂时不会复用：

```text
user input
constant
Graph output
retained saved Value
ExternalOp output
```

这不是 production allocator，而是一个可以逐行检查的 greedy educational buffer planner。

## 6. ExecutionPlan

```python
plan = build_execution_plan(
    lowered.program,
    retained_value_ids=saved_ids,
)

print(plan.format())
```

结构：

```text
ExecutionPlan
├── LoweredProgram
├── ValueLifetimeAnalysis
├── MemoryPlan
│   ├── BufferInterval[]
│   ├── BufferSlot[]
│   └── value → slot
├── ExecutionStep[]
└── release-after-node schedule
```

Memory metrics：

```text
logical reusable Value count
physical slot count
reuse_count
naive_allocation_bytes
planned_buffer_bytes
bytes_avoided_by_reuse
peak_eligible_live_bytes
```

这些是**结构/内存模型指标**，与 wall-clock latency 分开报告。

## 7. PlannedExecutor

```python
planned = PlannedExecutor(plan)
out = planned(x, y)
```

执行原则：

```text
PointwiseFusion
→ 可写入预分配 Storage slot

ExternalOp
→ 继续通过 Dispatcher
```

也就是说 v0.15 没有为了 buffer reuse 绕开 Dispatcher 或重写 matmul/sum 等 kernel。

每次 executor call 都创建独立 slot pool，所以一次运行返回的 Tensor 不会因为下一次运行复用 slot 而被覆盖。

## 8. AOT forward/backward 集成

v0.13/v0.14 的 saved-value ABI 继续成立：

```text
AOT forward Graph
      ↓
lower_graph
      ↓
map source saved Value ids
      ↓
build_execution_plan(retained=saved ids)
      ↓
PlannedExecutor.run_with_values
      ↓
saved Values
      ↓
planned backward Graph
      ↓
input gradients
```

验证链：

```text
eager Autograd gradients
      ==
AOT reference gradients
      ==
lowered backward gradients
      ==
planned backward gradients
```

## 9. Capture identity robustness fix

v0.15 的长表达式/大量临时 Tensor 测试暴露了一个 inherited compiler bug：旧 capture 只保存 `id(tensor) → Value`，短生命周期 Tensor 被 GC 后 Python 可能复用同一个 object id，导致两个不同 eager temporary 被错误映射为同一 Graph Value。

现在 capture session 在 capture 生命周期内保存强引用，并要求：

```text
same Python id
+
object identity is the same object
```

才能复用 Graph Value。

这保证一个 Graph Value 不会出现多个 producer。

## 10. 推荐阅读顺序

```text
mini_pytorch/compiler/benchmark.py
      ↓
mini_pytorch/compiler/profiler.py
      ↓
mini_pytorch/compiler/runtime_analysis.py
      ↓
mini_pytorch/compiler/execution_plan.py
      ↓
tests/test_runtime_performance.py
      ↓
tests/test_execution_plan.py
      ↓
docs/RUNTIME_PERFORMANCE.md
      ↓
examples/18_runtime_performance.py
```

同时可对照：

```text
v0.14 lowering.py
      ↓
v0.15 execution_plan.py
```

理解 compiler/runtime handoff。

## 11. Quick run

```bash
python examples/18_runtime_performance.py
python -m unittest discover -s tests -v
```

## 12. 明确非目标

v0.15 不实现：

- production caching allocator；
- CUDA/Triton/C++ codegen；
- GPU benchmark；
- async streams/events；
- multithreaded scheduler；
- symbolic-shape memory planner；
- arbitrary alias/mutation optimization；
- benchmark cherry-picking；
- “一定比 eager 更快”的结论。

## 13. 从 v0.1 到 v0.15

```text
v0.1   Dynamic Autograd
  ↓
v0.2   Optimizer / Scheduler
  ↓
v0.3   DataLoader
  ↓
v0.4   Storage / Views
  ↓
v0.5   Dispatcher
  ↓
v0.6   CNN operators
  ↓
v0.7   Device / Backend
  ↓
v0.8   Autograd Engine
  ↓
v0.9   Training Runtime
  ↓
v0.10  DDP
  ↓
v0.11  Graph IR
  ↓
v0.12  Guards / Cache
  ↓
v0.13  AOT backward Graph
  ↓
v0.14  Lowering / Fusion
  ↓
v0.15  Runtime / Performance Mechanics
  ↓
v1.0   Integrated Architecture
```

## 14. Next

v0.15 完成后不再继续堆新大机制，下一阶段进入 **v1.0 Integrated Architecture**：统一架构文档、端到端学习路径、API consistency、完整示例与最终 release snapshot。
