# mini-pytorch v0.7

## 1. Version identity

- **Branch:** `v0.7`
- **Previous version:** `v0.6`
- **Theme:** Device / Backend abstraction
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.7.0`
- **Validated regression baseline:** **103/103 tests passed**

v0.7 answers:

> How can Tensor allocation and operator execution carry explicit device/backend identity without scattering `if device == ...` throughout the framework?

## 2. Core difference from v0.6

v0.6 established:

```text
Module → functional op → Dispatcher → Autograd wrapper → CPU kernel
```

v0.7 adds placement:

```text
Tensor
  ├── shape / stride / dtype
  └── device
       ↓
Storage
  └── backend-owned allocation
       ↓
Dispatcher
  ├── CPU
  └── EDU simulated backend
```

`edu` is a deterministic educational backend that uses host NumPy memory internally. It exists to validate multi-backend architecture on ordinary CI and is not CUDA/GPU support.

## 3. Implemented mechanisms

### Device

- immutable `Device` value object
- `cpu` / `edu`
- optional non-negative index syntax
- equality/hash/string representation
- validation

### Backend registry

- `Backend` abstraction
- `allocate(size, dtype)`
- `from_numpy(array)`
- `to_numpy(allocation)`
- `NumPyBackend`
- `EducationalBackend`
- `backend_for()` / `list_backends()`

### Storage

Storage now owns:

```text
Device
Backend
backend-owned allocation
VersionCounter
```

and exposes `.device` / `.backend`.

### Tensor placement

- Tensor constructor accepts `device=`
- `.device`
- repr includes device
- views preserve Storage/device
- direct `Function.apply` outputs preserve common input device
- mixed-device direct Function inputs fail
- scalar promotion follows owning Tensor device

### Tensor.to

Semantics:

```text
same device + copy=False → same Tensor
same device + copy=True  → new Storage
cross-device             → new target Storage
```

Cross-device transfer materializes logical values and breaks aliasing.

When grad mode is enabled, transfer attaches an identity-gradient CopyToDevice node so gradients flow back to the source Tensor.

### Device-aware Dispatcher

New backend key:

```text
DispatchKey.EDU
```

Routes:

```text
CPU + grad  → {Autograd, CPU}
EDU + grad  → {Autograd, EDU}
CPU no_grad → {NoGrad, CPU}
EDU no_grad → {NoGrad, EDU}
```

Mixed-device Tensor operands fail before backend kernel execution.

### Backend registrations

CPU and EDU registrations exist for:

- add
- mul
- matmul
- relu
- sum
- conv2d
- max_pool2d

The educational backend intentionally reuses NumPy numerical implementations while exercising a distinct backend route.

### Module.to

- recursively moves Parameters
- recursively moves Buffers
- recursively moves child Modules
- preserves registered Parameter object identity
- Parameters remain leaves

### Functional device inheritance

- implicit Conv2d bias is created on weight.device
- Dropout mask is created on input.device
- BatchNorm validates Parameter/Buffer device consistency

## 4. New files

```text
mini_pytorch/device.py
mini_pytorch/backend.py
tests/test_device_backend.py
docs/DEVICE_BACKEND.md
examples/10_device_backend.py
```

Major modified files:

```text
mini_pytorch/storage.py
mini_pytorch/tensor.py
mini_pytorch/autograd.py
mini_pytorch/dispatcher.py
mini_pytorch/operator_registry.py
mini_pytorch/functional.py
mini_pytorch/nn/module.py
mini_pytorch/__init__.py
pyproject.toml
tests/test_dispatcher.py
README.md
```

## 5. Validation

Validated release-candidate commit:

```text
500debd4769108d1739b8afd6888fcd116fe5a1d
```

GitHub Actions environment:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

Result:

```text
Ran 103 tests in 0.663s
OK
```

v0.7-specific coverage includes:

- Device parse/equality/hash/errors
- backend registry/allocation
- Storage.device/backend
- Tensor.device
- same-device no-op/copy transfer
- cross-device copy and alias separation
- differentiable device transfer
- CPU vs EDU dispatcher selection
- Autograd/NoGrad routing above EDU
- mixed-device errors
- scalar promotion on EDU
- direct Function.apply device preservation
- Module.to Parameter/Buffer movement
- Conv2d and MaxPool2d execution on EDU

## 6. Definition of Done

- [x] Device construction/equality/validation
- [x] Tensor and Storage expose device identity
- [x] backend allocation boundary is explicit
- [x] CPU behavior remains compatible
- [x] Dispatcher inference uses backend identity
- [x] same operator API selects CPU vs EDU registration
- [x] mixed-device operations fail clearly
- [x] `.to()` no-op/copy/cross-device semantics are documented and tested
- [x] transfer preserves Autograd connectivity
- [x] views lose aliasing after cross-device transfer
- [x] direct Function.apply operators preserve device
- [x] Module.to moves Parameters/Buffers
- [x] Conv2d/MaxPool work through EDU route
- [x] full regression suite passes
- [x] architecture docs exist
- [x] educational example exists
- [x] package version is `0.7.0`
- [x] branch ready for exact final-head CI and integration into main

## 7. Recommended reading order

```text
mini_pytorch/device.py
mini_pytorch/backend.py
mini_pytorch/storage.py
mini_pytorch/tensor.py
mini_pytorch/dispatcher.py
mini_pytorch/operator_registry.py
mini_pytorch/nn/module.py
tests/test_device_backend.py
docs/DEVICE_BACKEND.md
examples/10_device_backend.py
```

## 8. Explicit non-goals

v0.7 intentionally does not implement:

- real CUDA
- CuPy
- asynchronous copies
- pinned memory
- streams/events
- mixed precision
- caching allocator
- distributed execution
- compiler graph capture

## 9. Learning goals

After v0.7 the reader should be able to explain:

1. Device vs Backend vs Storage vs DispatchKey;
2. why Tensor.device belongs in metadata but allocation belongs to a backend;
3. why cross-device transfer must allocate new Storage;
4. why transfer breaks aliasing but can preserve gradient connectivity;
5. why mixed-device math should fail instead of implicitly copy;
6. why Autograd is above backend dispatch rather than embedded inside kernels;
7. why a simulated second backend is enough to validate architecture before CUDA.

## 10. Next version

`v0.8` should focus on **Advanced Autograd Engine** concepts:

```text
graph lifetime
retain_grad
hooks
retain_graph
custom Function
higher-order-gradient foundations
```
