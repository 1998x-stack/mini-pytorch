# mini-pytorch v0.4

## 1. Version identity

- **Branch:** `v0.4`
- **Previous version:** `v0.3`
- **Theme:** TensorImpl-like memory model — Storage, stride, views, contiguous layout, inplace safety
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.4.0`

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

The primary question answered by this version is:

> How can two tensors have different shapes/layouts while referring to the same underlying memory, and why does that matter for Autograd?

---

## 2. Core difference from v0.3

### v0.3 mental model

```text
Tensor ≈ ndarray + requires_grad + grad_fn
```

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

A transpose, slice or compatible view can create a new Tensor object without allocating new Storage.

---

## 3. Mechanisms added

### Storage

- explicit `Storage`
- flat contiguous allocation
- dtype / size / nbytes metadata
- shared `VersionCounter`
- contiguous stride calculation

### Tensor layout metadata

- explicit shape
- explicit element strides
- explicit `storage_offset`
- `stride()`
- `is_contiguous()`
- correct zero-dimensional scalar layout

### Views

- `transpose(dim0, dim1)`
- `permute(*dims)`
- basic slicing views
- negative-stride slices when representable
- Storage alias preservation
- backward through view operations

### `view` / `reshape` / `contiguous`

- `view()` aliases Storage or fails; it never silently copies
- `reshape()` aliases when possible and copies when required
- `contiguous()` returns self for an already-contiguous Tensor and materializes otherwise

### Inplace / Autograd safety

- tracked `add_`, `mul_`, `copy_`
- leaf `requires_grad=True` inplace restriction while grad is enabled
- shared Storage version counter across aliases
- optimizer updates bump parameter versions
- `load_state_dict()` bumps destination versions
- `Context.save_for_backward()` records versions of saved input values
- backward validates saved versions and rejects invalidated values

---

## 4. Observable invariants

```text
contiguous tensor -> canonical element strides
transpose         -> same Storage, different stride
permute           -> same Storage, reordered stride
slice             -> same Storage, changed storage_offset/stride
negative slice    -> negative stride
view              -> must alias or fail
reshape           -> alias when possible, copy when required
contiguous        -> allocate only when required
inplace alias     -> all aliases see mutation + shared version bump
optimizer update  -> parameter version bump
state loading     -> destination version bump
saved mutation    -> backward RuntimeError
```

---

## 5. What remains unchanged from v0.3

The higher-level training stack is intentionally preserved:

- eager dynamic Autograd graph
- Module / Parameter registration
- Linear / activations / losses
- optimizer families
- LR schedulers
- Dataset / Sampler / BatchSampler / DataLoader
- MNIST classification path

v0.4 changes Tensor memory semantics underneath those layers rather than redesigning them.

---

## 6. Important files

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

1. why Tensor is not synonymous with its allocation;
2. what strides mean in element units;
3. how `storage_offset` represents a slice into shared Storage;
4. why transpose/permute can be metadata-only operations;
5. why non-contiguous tensors are still valid tensors;
6. why `view()` may fail while `reshape()` succeeds;
7. what `contiguous()` materializes;
8. why aliases need a shared mutation version counter;
9. why Autograd validates saved values before backward;
10. why optimizer/state-loading writes participate in the same mutation model.

---

## 8. Validation record

### Exact remote validation

The exact v0.4 code was validated by GitHub Actions on:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

Validated implementation commit:

```text
e12c947a1d6e6f44779e7f2bb8ff9b776a80f808
```

Result:

```text
Ran 65 tests in 0.510s
OK
```

The subsequent `version.md`-only release-candidate commit was also revalidated successfully before integration.

### Validation bug found and fixed

Initial CI exposed that `np.ascontiguousarray()` promotes a 0-D ndarray to shape `(1,)`. That would have changed scalar Tensor semantics. v0.4 fixes this by materializing contiguous Storage while preserving the original scalar shape `()`.

---

## 9. Definition of Done

- [x] explicit `Storage`
- [x] shape / stride / storage offset metadata
- [x] transpose / permute / basic slicing aliases
- [x] shared-storage invariant tests
- [x] distinct `view()` / `reshape()` semantics
- [x] `contiguous()` semantics
- [x] backward through view operations
- [x] shared version counter
- [x] tracked inplace operations
- [x] optimizer/state-loading version integration
- [x] saved-tensor version mismatch detection
- [x] zero-dimensional scalar compatibility
- [x] architecture documentation
- [x] educational example
- [x] package version `0.4.0`
- [x] exact remote CI: **65/65 tests pass**
- [x] validated implementation synchronized into `main`

---

## 10. Explicit non-goals

v0.4 intentionally excludes:

- operator registry / Dispatcher
- device/backend dispatch
- CUDA
- advanced indexing parity
- complete alias analysis
- channels-last / arbitrary memory formats
- sparse / quantized storage
- production storage resizing
- full PyTorch inplace API parity

These belong to later architectural milestones.

---

## 11. Next version: v0.5

`v0.5` introduces an operator registry and Dispatcher-like routing layer.

```text
v0.4
Storage + Tensor layout semantics
        ↓
v0.5
operator schema / registry / dispatch key / kernel routing
        ↓
later versions
CPU / accelerator / Autograd-specific implementations
```

The dependency order is deliberate: first define what a Tensor is, then define how operations on that Tensor are routed.