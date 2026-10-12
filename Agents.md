# Agents.md — mini-pytorch v0.4

> Read this file before modifying this branch.

## Branch identity

- Branch: `v0.4`
- Role: **frozen historical learning snapshot**
- Previous version: `v0.3`
- Theme: Storage + stride + storage_offset + views + contiguous layout + inplace/autograd safety
- Required companion document: `version.md`

## Mandatory rule

v0.4 is complete and frozen. It exists to preserve the Tensor memory-model milestone and must not absorb later-version architecture.

Allowed changes after freeze:

- documentation corrections;
- test corrections that preserve v0.4 semantics;
- narrowly scoped bug fixes required for behavior already promised by `version.md`.

Any semantic bug fix to frozen v0.4 must also be reflected in `main` when still applicable.

## Required reading order

1. `Agents.md`
2. `version.md`
3. `mini_pytorch/storage.py`
4. `mini_pytorch/tensor.py`
5. `mini_pytorch/autograd.py`
6. `mini_pytorch/ops.py`
7. `tests/test_storage.py`
8. `docs/TENSOR_MEMORY.md`

## Preserved mental model

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

## v0.4 scope

This branch owns and preserves:

- explicit `Storage`;
- contiguous stride calculation;
- Tensor shape/stride/storage_offset metadata;
- transpose / permute;
- basic slicing views;
- `is_contiguous()`;
- `view()` / `reshape()` distinction;
- `contiguous()`;
- shared Storage aliasing;
- backward through view operations;
- shared version counter;
- tracked educational inplace operations;
- saved-tensor version validation;
- optimizer/state-loading version coherence.

## Forbidden future scope

Do not add:

- operator registry / Dispatcher (`v0.5`);
- Conv2d / pooling / BatchNorm (`v0.6`);
- Device/backend/CUDA abstraction (`v0.7`);
- advanced Autograd graph-lifetime features (`v0.8`);
- DDP / collectives;
- graph capture / compiler;
- unrelated API-completeness work.

## Storage invariants

1. Storage owns a flat contiguous allocation.
2. Tensor layout metadata is independent of Storage allocation metadata.
3. Metadata-only views reuse Storage.
4. Copies receive independent Storage/version state.
5. Aliases observe one shared mutation version.
6. `storage_offset` and strides use element units.
7. Zero-dimensional scalar Tensors preserve shape `()`.

## View invariants

- `transpose` / `permute` are metadata-only aliases where representable;
- basic slicing preserves Storage where possible;
- `view()` fails rather than silently copying;
- `reshape()` may alias or copy;
- `contiguous()` materializes only when required.

## Autograd / mutation invariants

- only input values actually saved for backward require saved-version validation;
- tracked inplace mutation through any alias bumps the shared version counter;
- mutation of a saved value after forward raises during backward;
- leaf requires-grad tracked inplace operations require `no_grad()`;
- optimizer and state-loading writes participate in versioning.

## Validation record

The exact v0.4 implementation passed the remote regression suite:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
Ran 65 tests
OK
```

## Next-version boundary

`v0.5` owns operator schema/registry/Dispatcher concepts. Do not backport those concepts into this branch.