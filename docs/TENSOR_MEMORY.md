# Tensor Memory Model — mini-pytorch v0.4

v0.4 replaces the earlier mental model

```text
Tensor = NumPy array + autograd metadata
```

with a more PyTorch-like model:

```text
Tensor = Storage reference + layout metadata + autograd metadata
```

The purpose is not to reproduce every detail of `TensorImpl`; it is to expose the mechanisms that explain `view`, `reshape`, `transpose`, `contiguous`, aliasing and inplace-autograd errors.

---

## 1. Storage versus Tensor

`Storage` owns one flat contiguous allocation:

```text
Storage
index:  0  1  2  3  4  5
value:  a  b  c  d  e  f
```

A Tensor describes how to interpret that allocation:

```text
Tensor
├── shape
├── stride
├── storage_offset
├── dtype
└── Storage reference
```

Two Tensors may therefore share the same Storage.

```python
x = Tensor([[0, 1, 2],
            [3, 4, 5]])
y = x.T

assert x.storage is y.storage
```

No element data needs to be copied for this transpose.

---

## 2. Strides are measured in elements

For a contiguous `(2, 3)` tensor:

```text
shape  = (2, 3)
stride = (3, 1)
```

Moving one position along dimension 0 advances three elements in Storage. Moving one position along dimension 1 advances one element.

The address calculation is conceptually:

```text
storage_index = storage_offset
              + i0 * stride[0]
              + i1 * stride[1]
              + ...
```

For a transpose:

```text
original shape  = (2, 3)
original stride = (3, 1)

transpose shape  = (3, 2)
transpose stride = (1, 3)
```

The data allocation is unchanged; only metadata changes.

---

## 3. storage_offset explains slicing

Given:

```python
x = Tensor(np.arange(12).reshape(3, 4))
y = x[1:, 1:]
```

`y` can reuse `x.storage` and begin at element 5:

```text
y.storage is x.storage

y.storage_offset == 5
y.stride()       == (4, 1)
```

A negative-step slice can also be represented with a negative stride:

```python
y = x_flat[::-1]

storage_offset = last element index
stride         = (-1,)
```

---

## 4. Contiguous does not mean “has data”

A tensor is contiguous when its logical dimensions follow the canonical row-major stride pattern.

```python
x = Tensor(np.arange(6).reshape(2, 3))
assert x.is_contiguous()

xt = x.T
assert not xt.is_contiguous()
```

`xt` still has perfectly valid data. Non-contiguous only means its logical traversal order does not match canonical row-major layout.

---

## 5. view versus reshape

### `view()`

`view()` promises a metadata-only reinterpretation.

```python
x = Tensor(np.arange(12).reshape(3, 4))
y = x.view(2, 6)

assert y.storage is x.storage
```

If a layout cannot be represented without copying, `view()` fails.

```python
x = Tensor(np.arange(6).reshape(2, 3)).T
x.view(6)  # RuntimeError
```

### `reshape()`

`reshape()` asks for a shape, not for guaranteed aliasing.

It behaves conceptually as:

```text
can reshape be represented as a view?
        │
   yes  │  no
        │
        ▼
 share Storage     materialize a copy
```

Therefore:

```python
x = Tensor(np.arange(6).reshape(2, 3)).T
y = x.reshape(6)

assert y.storage is not x.storage
```

for an incompatible non-contiguous layout.

---

## 6. contiguous()

`contiguous()` makes the requested logical values live in canonical contiguous layout.

```python
x = Tensor(np.arange(6).reshape(2, 3))
assert x.contiguous() is x

xt = x.T
c = xt.contiguous()

assert c.storage is not xt.storage
assert c.is_contiguous()
```

This is why calling `contiguous()` can allocate memory.

---

## 7. Why aliasing changes inplace semantics

Suppose:

```python
x = Tensor(np.arange(6).reshape(2, 3))
y = x.T
```

`x` and `y` share Storage. Therefore a mutation through `y` changes values visible through `x`.

```python
y.add_(10)
```

This is not two independent tensors anymore. They are two views over one allocation.

---

## 8. Shared version counter

Every Storage owns one mutation counter.

```text
Storage
├── allocation
└── version = N
      ▲
      ├── Tensor A
      └── Tensor B
```

Tracked inplace operations bump that counter:

```python
before = x._version
y.add_(1)
assert x._version == before + 1
assert y._version == x._version
```

Optimizers and `load_state_dict()` also mark the underlying Tensor dirty because they mutate parameter/buffer values.

---

## 9. Why Autograd cares about versions

Consider:

```python
x = Tensor([2.0, 3.0], requires_grad=True)
y = (x * x).sum()
```

The derivative of multiplication needs the input values. The forward pass therefore saves values required for backward.

Now mutate an alias:

```python
alias = x.detach()
alias.add_(1)
```

If backward silently reused the modified values, it could compute the wrong gradient.

v0.4 stores the expected Storage version for input values that are saved for backward. Before the backward function runs:

```text
saved version == current version ?
        │
   yes  │  no
        │
        ▼
 run backward      raise RuntimeError
```

This directly explains the class of PyTorch errors that say a variable needed for gradient computation was modified by an inplace operation.

---

## 10. Why not version-check every operator input?

That would be too strict.

For example, addition backward only needs the output gradient and original shapes; it does not require the original numerical input values.

The educational rule is:

> only tensors whose values are actually saved for backward participate in saved-version validation.

This preserves the distinction between graph connectivity and saved data dependencies.

---

## 11. `.data` is intentionally an unsafe escape hatch

The framework still exposes NumPy arrays through `tensor.data` because earlier versions and optimizer implementations use it for transparent educational code.

Direct external mutation such as:

```python
x.data[...] = 0
```

can bypass tracked Tensor inplace policy unless framework code explicitly calls `_mark_dirty()`.

Use `add_`, `mul_`, `copy_`, optimizers, or state-loading APIs when demonstrating version-aware mutation.

This limitation is intentional and documented; v0.4 teaches the mechanism rather than building a protected ndarray proxy system.

---

## 12. Mapping to real PyTorch concepts

mini-pytorch v0.4 conceptually corresponds to a small subset of ideas surrounding real PyTorch Tensor metadata:

```text
mini-pytorch                  real PyTorch concept
---------------------------------------------------------------
Storage                       Storage / storage allocation
Tensor._shape                 sizes
Tensor._strides               strides
Tensor._storage_offset        storage_offset
Tensor._version               version counter
view aliases                  Tensor views / aliasing
saved version checks          inplace autograd correctness
```

The implementation is intentionally original and much smaller. The goal is transfer of the mental model, not source compatibility.

---

## 13. Reading order

Read v0.4 in this order:

```text
storage.py
   ↓
tensor.py
   ↓
ops.py
   ↓
autograd.py
   ↓
tests/test_storage.py
   ↓
examples/07_tensor_memory.py
```

Then compare the same concepts with real PyTorch using small experiments involving `stride()`, `storage_offset()`, `is_contiguous()`, `view()`, `reshape()` and inplace-autograd errors.