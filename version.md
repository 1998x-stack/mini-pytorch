# mini-pytorch v0.4

## 1. Version identity

- **Branch:** `v0.4`
- **Base:** latest integrated `main` after the v0.3 data-pipeline milestone
- **Previous version:** `v0.3`
- **Theme:** TensorImpl-like memory model — Storage, stride, views, contiguous layout, inplace safety
- **Status:** release candidate — implementation complete, awaiting final remote CI and `main` synchronization

v0.4 changes the meaning of a Tensor more deeply than any previous release. In v0.1-v0.3, a Tensor was effectively a NumPy array plus autograd metadata. In v0.4, Tensor identity is decomposed into an allocation plus view metadata.

```text
Storage
  └── flat contiguous allocation
          ▲
          │ shared by aliases
   ┌──────┴────────┐
   │               │
Tensor A        Tensor B
shape           shape
stride          stride
storage_offset  storage_offset
```

The primary question for this version is:

> How can two tensors have different shapes/layouts while referring to the same underlying memory, and why does that matter for Autograd?

---

## 2. Core difference from v0.3

### v0.3 mental model

```text
Tensor ≈ ndarray + requires_grad + grad_fn
```

NumPy implicitly owns most memory-layout behavior.

### v0.4 mental model

```text
Tensor
├── Storage reference
├── shape
├── stride
├── storage_offset
├── dtype
├── requires_grad
└── grad_fn
```

A transpose, slice or compatible view can now create a new Tensor object without allocating a new Storage.

---

## 3. Mechanisms added in v0.4

### Storage

- explicit `Storage`
- flat contiguous allocation
- dtype and size metadata
- shared version counter
- contiguous stride calculation

### Tensor layout

- explicit shape
- explicit element strides
- explicit `storage_offset`
- `stride()`
- `is_contiguous()`

### Views

- `transpose(dim0, dim1)`
- `permute(*dims)`
- basic slicing views, including negative strides where NumPy can represent a view
- Storage alias preservation
- backward through view operations

### view / reshape / contiguous

- `view()` requires a metadata-only compatible layout
- `reshape()` returns a view when possible and copies when required
- `contiguous()` returns self when already contiguous and materializes a contiguous copy otherwise

### Inplace/autograd safety

- shared Storage version counter
- tracked educational inplace ops: `add_`, `mul_`, `copy_`
- version bump visible through all aliases
- optimizer updates bump parameter Storage versions
- `load_state_dict()` bumps target Storage versions
- saved-tensor version validation before backward
- error when a saved value is changed after forward
- leaf-requires-grad inplace restriction unless grad is disabled

---

## 4. What remains unchanged from v0.3

v0.4 preserves the higher-level training stack:

- eager dynamic Autograd graph
- Module / Parameter registration
- Linear / activation / loss modules
- optimizer families and LR schedulers
- Dataset / Sampler / BatchSampler / DataLoader
- MNIST classification example

This version changes Tensor memory semantics underneath those layers instead of redesigning them.

---

## 5. Demonstrable behavior

The implementation and tests make these statements observable:

```text
transpose shares Storage
transpose can be non-contiguous
permute changes strides without copying
slice can change storage_offset without copying
negative-step slice can use a negative stride
view can fail on incompatible layout
reshape may allocate new Storage
contiguous may allocate new Storage
aliases observe tracked inplace mutation
aliases share one version counter
optimizer/state loading participate in versioning
saved-value mutation can invalidate backward
```

---

## 6. Important source files

Recommended reading order:

```text
mini_pytorch/storage.py
        ↓
mini_pytorch/tensor.py
        ↓
mini_pytorch/autograd.py
        ↓
mini_pytorch/ops.py
        ↓
tests/test_storage.py
        ↓
docs/TENSOR_MEMORY.md
        ↓
examples/07_tensor_memory.py
```

---

## 7. Learning goals

After v0.4, the reader should be able to explain:

1. why Tensor is not synonymous with its underlying allocation;
2. what a stride means in element units;
3. how `storage_offset` lets a slice start inside shared Storage;
4. why transpose can be O(1) metadata manipulation;
5. why a transposed tensor is commonly non-contiguous;
6. why `view()` may fail while `reshape()` succeeds;
7. what `contiguous()` actually materializes;
8. why aliases need a shared mutation/version counter;
9. why Autograd must detect mutation of values saved for backward;
10. why parameter/state mutations must participate in the same versioning model.

---

## 8. Explicit non-goals

v0.4 intentionally does **not** implement:

- a Dispatcher or operator registry;
- device/backend dispatch;
- CUDA;
- arbitrary advanced NumPy indexing parity;
- a complete alias-analysis system;
- all PyTorch memory formats;
- channels-last layout;
- sparse/quantized storage;
- production-grade storage resizing;
- full PyTorch inplace API parity.

Those would obscure the Storage/stride/view mental model.

---

## 9. Validation status

A locally reconstructed v0.4 core was executed with the existing regression tests plus the new memory-model tests before the remote commits were finalized. That validation passed.

The repository also now contains GitHub Actions CI so the exact remote branch can be validated with:

```bash
python -m unittest discover -s tests -v
```

Do not treat v0.4 as frozen until the latest remote CI run succeeds and the exact validated branch state is synchronized to `main`.

---

## 10. Definition of Done

- [x] `Storage` is explicit and independently inspectable.
- [x] Tensor exposes shape, stride and storage offset.
- [x] transpose / permute / basic slicing preserve Storage when appropriate.
- [x] shared-storage aliasing has invariant tests.
- [x] `view` / `reshape` differ observably on non-contiguous input.
- [x] `contiguous()` behavior is tested.
- [x] view backward maps gradients correctly.
- [x] tracked inplace ops bump the shared version counter.
- [x] optimizer and state-loading writes bump versions.
- [x] saved-tensor version mismatch is detected before backward.
- [x] educational example exists.
- [x] architecture documentation exists.
- [x] package version is `0.4.0`.
- [ ] latest exact remote branch passes CI.
- [ ] validated state is synchronized to `main`.

---

## 11. Next version

`v0.5` introduces an operator registry and Dispatcher-like routing layer.

The dependency rationale is deliberate:

```text
v0.4: define what a Tensor/storage/layout is
                  ↓
v0.5: define how Tensor operations are registered and routed
                  ↓
later: select CPU / Autograd / accelerator implementations
```

This avoids scattering backend conditionals across Tensor methods.