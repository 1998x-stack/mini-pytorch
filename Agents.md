# Agents.md — mini-pytorch v0.4

> Read this file before modifying this branch.

## Branch identity

- Branch: `v0.4`
- Role: active Tensor memory-model learning snapshot
- Previous version: `v0.3`
- Theme: Storage + stride + storage_offset + views + contiguous layout + inplace/autograd safety
- Required companion document: `version.md`

## Mandatory reading order

1. `Agents.md`
2. `version.md`
3. `mini_pytorch/storage.py`
4. `mini_pytorch/tensor.py`
5. `mini_pytorch/autograd.py`
6. `mini_pytorch/ops.py`
7. `tests/test_storage.py`
8. relevant regression tests

## Primary mental model

Every change on this branch must reinforce this model:

```text
Storage allocation
      ▲
      │ shared
┌─────┴──────┐
Tensor A   Tensor B
shape      shape
stride     stride
offset     offset
```

A Tensor is a logical view over Storage, not merely an ndarray wrapper.

## Allowed scope

- explicit `Storage`
- contiguous stride calculation
- Tensor shape/stride/storage_offset metadata
- transpose / permute
- basic slicing views
- `is_contiguous()`
- `view()` / `reshape()` distinction
- `contiguous()`
- Storage aliasing tests
- backward through view operations
- shared version counter
- tracked educational inplace ops
- saved-tensor version validation
- minimal changes to Module/Optimizer code needed to keep version counters coherent
- tests/docs/examples required to explain those mechanisms

## Forbidden scope

Do not implement future-version systems here:

- operator registry / Dispatcher (`v0.5`)
- Conv2d / pooling / BatchNorm (`v0.6`)
- Device/backend/CUDA abstraction (`v0.7`)
- retain_graph/hooks/higher-order Autograd (`v0.8`)
- DDP / collectives
- graph capture / compiler

Do not add advanced indexing merely to chase NumPy/PyTorch API completeness.

## Storage rules

1. Storage owns a flat contiguous allocation.
2. Tensor view metadata is independent from Storage allocation metadata.
3. Views that can be represented by metadata must reuse the same Storage.
4. Copies must receive a new Storage/version counter.
5. All aliases of one Storage observe the same tracked mutation version.
6. `storage_offset` and strides are measured in elements, not bytes.
7. NumPy is a reference mechanism, not the conceptual source of truth.

## View rules

- `transpose`/`permute` should normally be metadata-only aliases.
- basic slicing should preserve Storage when NumPy basic indexing can represent a view.
- `view()` must fail instead of silently copying when layout is incompatible.
- `reshape()` may return a view or a copy.
- `contiguous()` may return the same Tensor when already contiguous; otherwise it materializes a new Storage.

## Autograd/inplace rules

- backward must validate versions for input values actually saved for backward.
- an inplace write through any tracked alias must bump the shared Storage version.
- mutation of a saved value after forward must raise a clear RuntimeError during backward.
- a leaf Tensor requiring grad cannot use tracked inplace Tensor ops while grad is enabled; `no_grad()` is the educational parameter-update escape hatch.
- do not over-reject operations whose backward does not depend on the original value.

## Testing contract

v0.4 is incomplete unless tests prove:

- contiguous stride values;
- transpose Storage identity;
- non-contiguous transpose layout;
- permute layout;
- slice storage_offset;
- contiguous `view` aliases Storage;
- incompatible non-contiguous `view` fails;
- non-contiguous `reshape` can allocate;
- `contiguous()` allocation behavior;
- shared alias version counter;
- inplace leaf restriction;
- saved-tensor version mismatch detection;
- gradients through transpose/reshape/slice;
- previous Autograd/NN/Optimizer/Data behavior still works.

Preferred test command:

```bash
python -m unittest discover -s tests -v
```

Do not report full-suite success unless the exact branch contents were executed.

## Documentation contract

Before freezing v0.4:

- update `version.md` to completed/frozen;
- add a focused tensor-memory architecture document;
- add an educational example;
- update package version to `0.4.0`;
- update `Plan.md` on `main` if status/sequencing changes;
- ensure validated v0.4 code is integrated into `main`.

## Next-version boundary

`v0.5` owns the operator registry/Dispatcher concept. v0.4 Tensor methods may still invoke operator classes directly. Do not prematurely create dispatch keys or backend registries here.