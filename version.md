# mini-pytorch v0.4

## 1. Version identity

- **Branch:** `v0.4`
- **Base:** latest integrated `main` after the v0.3 data-pipeline milestone
- **Previous version:** `v0.3`
- **Theme:** TensorImpl-like memory model — Storage, stride, views, contiguous layout, inplace safety
- **Status:** validated release candidate — remote CI green; integration into `main` pending

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
- dtype / size / nbytes metadata
- shared `VersionCounter`
- contiguous stride calculation

### Tensor layout metadata

- explicit shape
- explicit element strides
- explicit `storage_offset`
- `stride()`
- `is_contiguous()`
- zero-dimensional scalar layout preserved correctly

### View operations

- `transpose(dim0, dim1)`
- `permute(*dims)`
- basic slicing views
- negative-stride slice support where representable
- shared Storage alias preservation
- backward through transpose/reshape/slice

### `view` / `reshape` / `contiguous`

- `view()` requires a metadata-only compatible layout and fails rather than copying
- `reshape()` may share Storage or allocate a new Storage
- `contiguous()` returns self when already contiguous and materializes otherwise

### Inplace and Autograd safety

- tracked `add_`, `mul_`, `copy_`
- leaf `requires_grad=True` inplace restriction while grad is enabled
- shared Storage version counter visible through aliases
- optimizer updates bump parameter versions
- `load_state_dict()` bumps destination versions
- forward `Context` records versions of input values actually saved for backward
- backward checks saved versions and rejects invalidated values

---

## 4. What remains unchanged from v0.3

v0.4 intentionally preserves the higher-level stack:

- eager dynamic Autograd graph
- Module / Parameter registration
- Linear / activation / loss modules
- SGD / Adam / AdamW / RMSprop / Adagrad / Adamax
- LR schedulers
- Dataset / Sampler / BatchSampler / DataLoader
- MNIST classification path

The version changes Tensor memory semantics underneath those systems rather than redesigning them.

---

## 5. Observable invariants

The tests make the following behaviors explicit:

```text
contiguous tensor -> canonical element strides
transpose         -> same Storage, different stride
permute           -> same Storage, reordered stride
slice             -> same Storage, changed storage_offset/stride
negative slice    -> negative stride
view              -> must alias or fail
reshape           -> alias when possible, copy when required
contiguous        -> materialize only when required
inplace alias     -> all aliases see mutation + shared version bump
optimizer update  -> parameter version bump
state loading     -> destination version bump
saved mutation    -> backward RuntimeError
```

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
5. why non-contiguous tensors are valid tensors;
6. why `view()` may fail while `reshape()` works;
7. what `contiguous()` actually allocates;
8. why aliases need a shared version counter;
9. why Autograd must detect mutation of saved values;
10. why optimizer/state-loading writes belong to the same mutation model.

---

## 8. Validation

### Local development validation

The reconstructed development copy passed its regression + v0.4 memory-model tests after the scalar-layout fix.

### Exact remote branch validation

GitHub Actions validated the exact remote commit:

```text
e12c947a1d6e6f44779e7f2bb8ff9b776a80f808
```

Environment included:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

Result:

```text
Ran 65 tests in 0.510s
OK
```

This run covers the existing Autograd/NN/optimizer/scheduler/data/MNIST tests plus the new v0.4 Storage/layout/version tests.

---

## 9. Important bug found during validation

The first remote run exposed a real v0.4 regression: `np.ascontiguousarray()` promoted a 0-D ndarray to shape `(1,)`, turning scalar Tensor layout into one-dimensional layout.

The fix was architectural rather than test-specific:

```text
before: scalar Tensor shape () -> ascontiguousarray -> (1,)   WRONG
after:  scalar Tensor shape () -> np.array(order="C") -> ()  CORRECT
```

The final green CI confirms legacy scalar Autograd semantics remain intact.

---

## 10. Explicit non-goals

v0.4 intentionally does **not** implement:

- operator registry / Dispatcher
- device/backend dispatch
- CUDA
- advanced indexing parity
- complete alias analysis
- channels-last / arbitrary memory formats
- sparse / quantized storage
- production storage resizing
- full PyTorch inplace API parity

Those belong to later architectural milestones.

---

## 11. Definition of Done

- [x] explicit `Storage`
- [x] shape / stride / storage offset metadata
- [x] transpose / permute / basic slicing aliases
- [x] shared-storage invariant tests
- [x] distinct `view()` / `reshape()` behavior
- [x] `contiguous()` semantics
- [x] backward through view operations
- [x] shared version counter
- [x] tracked inplace operations
- [x] optimizer/state-loading version integration
- [x] saved-tensor version mismatch detection
- [x] zero-dimensional scalar compatibility
- [x] educational example
- [x] `docs/TENSOR_MEMORY.md`
- [x] package version `0.4.0`
- [x] exact remote CI: **65/65 tests pass**
- [ ] synchronize this validated snapshot into `main`

---

## 12. Next version: v0.5

`v0.5` owns the operator registry and Dispatcher-like routing layer.

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