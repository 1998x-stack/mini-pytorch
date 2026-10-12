# mini-pytorch v0.4 — Storage / Stride / Views / Inplace Safety

> 这一版开始真正回答：**Tensor 在内存里到底是什么？**

**版本主题：** TensorImpl-like memory model

**状态：** frozen historical learning snapshot

**Package version：** `0.4.0`

## 1. v0.4 相比 v0.3 的根本变化

在 v0.1-v0.3 中，可以近似理解为：

```text
Tensor ≈ NumPy ndarray + Autograd metadata
```

v0.4 把 Tensor 拆成：

```text
Tensor
├── Storage reference
├── shape
├── stride
├── storage_offset
├── requires_grad / grad_fn
└── view-specific metadata
```

多个 Tensor 可以共享同一个 Storage：

```text
                    Storage
          ┌────────────┴────────────┐
          │                         │
      Tensor A                  Tensor B
  shape=(2, 3)              shape=(3, 2)
  stride=(3, 1)             stride=(1, 3)
```

这正是理解真实 PyTorch `TensorImpl + Storage + sizes/strides/storage_offset` 的关键入口。

## 2. Storage

新增：

- explicit `Storage`
- flat contiguous allocation
- dtype
- shared `VersionCounter`
- `contiguous_strides()`

Tensor 的 `data` 现在是通过 Storage + metadata 重建出来的逻辑 view，而不是 Tensor 身份的全部。

## 3. Stride / storage_offset

```python
x = Tensor(np.arange(6).reshape(2, 3))
print(x.shape)      # (2, 3)
print(x.stride())   # (3, 1)

 y = x.T
print(y.shape)      # (3, 2)
print(y.stride())   # (1, 3)
```

`x` 和 `y` 可以共享 Storage，但拥有不同的逻辑布局。

## 4. View semantics

本版本支持：

- `transpose()` / `.T`
- `permute()`
- basic slicing view
- negative-step slicing
- `storage_offset`
- shared-storage aliasing

核心事实：

```text
view 不等于 copy
```

例如 transpose 通常只是修改 metadata。

## 5. `view()` vs `reshape()`

v0.4 将两者真正区分：

```text
view()
  ↓
只能 metadata-only reinterpretation
  ↓
不允许偷偷 copy
```

而：

```text
reshape()
   ├── layout 兼容 → view
   └── layout 不兼容 → materialize copy
```

因此：

```python
x = Tensor(np.arange(6).reshape(2, 3)).T
x.view(6)      # may fail
x.reshape(6)   # may allocate and succeed
```

## 6. contiguous

新增：

- `is_contiguous()`
- `contiguous()`

语义：

```text
already contiguous
      ↓
return self

non-contiguous
      ↓
logical-order copy
      ↓
new Storage
      ↓
canonical contiguous stride
```

## 7. Inplace + Version Counter

新增教学型 inplace operations：

- `add_()`
- `mul_()`
- `copy_()`

所有 alias 共享 Storage version：

```text
x ----┐
      ├── Storage(version=0)
alias-┘

alias.add_(1)
      ↓
Storage(version=1)
```

Autograd forward 保存依赖值时会同时保存 expected version。若 backward 前该 Storage 被修改：

```text
expected_version != current_version
```

则 backward 拒绝继续，演示真实 PyTorch inplace safety 的核心思想。

## 8. Framework mutation 也纳入 version 语义

不仅用户态 `add_()` 会 bump version，以下框架内部写入同样追踪：

- SGD
- Adam
- AdamW
- Adamax
- RMSprop
- Adagrad
- `load_state_dict()`

因此 optimizer update 与 Tensor inplace mutation 使用统一的内存版本模型。

## 9. 推荐源码阅读顺序

```text
mini_pytorch/storage.py
        ↓
mini_pytorch/tensor.py
        ↓
mini_pytorch/ops.py
        ↓
mini_pytorch/autograd.py
        ↓
tests/test_storage.py
        ↓
docs/TENSOR_MEMORY.md
        ↓
examples/07_tensor_memory.py
```

重点理解：

1. 一个 Tensor 为什么不等于一块独占内存；
2. stride 如何把 logical index 映射到 Storage；
3. transpose 为什么可以 O(1)；
4. non-contiguous 到底是什么意思；
5. `view` 为什么可能失败；
6. `reshape` 为什么可能 copy；
7. inplace 为什么会威胁 backward correctness；
8. alias 为什么必须共享 version counter。

## 10. 快速运行

```bash
pip install -e .
python -m unittest discover -s tests -v
python examples/07_tensor_memory.py
```

v0.4 frozen branch 已通过 GitHub Actions 全量测试。

## 11. 本版本不做什么

v0.4 仍然刻意不引入：

- operator registry
- Dispatcher / DispatchKey
- device/backend abstraction
- Conv2d
- advanced Autograd graph lifetime
- DDP
- graph compiler

因为在进入多 backend / Autograd routing 之前，必须先把 Tensor memory semantics 建立清楚。

## 12. 下一版本

`v0.5` 将引入 PyTorch 软件架构里极重要的一层：

```text
Tensor API
   ↓
Operator Schema
   ↓
Registry
   ↓
Dispatcher
   ↓
DispatchKeySet
   ├── Autograd
   ├── CPU
   ├── NoGrad
   └── future accelerator
```

目标是解释：**为什么一个 `x + y` 不应该在 Tensor 类里硬编码所有 backend/autograd 行为。**

详细版本边界见 [`version.md`](./version.md)，内存模型深度说明见 [`docs/TENSOR_MEMORY.md`](./docs/TENSOR_MEMORY.md)。