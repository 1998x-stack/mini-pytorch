# mini-pytorch v0.7

## 1. Version identity

- **Branch:** `v0.7`
- **Previous version:** `v0.6`
- **Theme:** Device / Backend abstraction
- **Status:** active until exact final-head CI is green
- **Target package version:** `0.7.0`

v0.7 answers one question:

> How can Tensor allocation and operator execution carry explicit device/backend identity without scattering `if device == ...` throughout the framework?

## 2. Core delta from v0.6

v0.6 established:

```text
Module → functional op → Dispatcher → Autograd wrapper → CPU kernel
```

v0.7 adds an explicit placement dimension:

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

`edu` is a deterministic educational backend. It intentionally uses NumPy host arrays internally so ordinary CI can validate multiple backend identities without GPU hardware.

## 3. Required implementation

- `Device` value object and validation
- backend registry
- CPU backend
- simulated `edu` backend
- Storage records device/backend identity
- Tensor exposes `.device`
- Tensor construction accepts `device=`
- `.to(device, copy=False)`
- cross-device transfer creates new Storage
- same-device no-op unless `copy=True`
- transfer participates in Autograd as an identity-gradient copy
- Dispatcher infers backend DispatchKey from Tensor device
- explicit `EDU` DispatchKey
- mixed-device operations fail before backend math
- scalar promotion follows the owning Tensor device
- dispatched arithmetic / Conv2d / MaxPool2d can route to the simulated backend
- direct `Function.apply` operators preserve common input device

## 4. Explicit non-goals

- real CUDA kernels
- CuPy dependency
- asynchronous copies
- pinned memory
- streams/events
- mixed precision
- production allocator semantics
- distributed execution

## 5. Definition of Done

- [ ] Device equality/hash/validation tests
- [ ] Storage/Tensor expose device
- [ ] backend allocation boundary is explicit
- [ ] `.to()` copy/no-op semantics tested
- [ ] transfer keeps gradients connected
- [ ] Dispatcher chooses CPU vs EDU from device
- [ ] mixed-device operators fail clearly
- [ ] same operator API executes on two backend registrations
- [ ] direct Autograd operators preserve device
- [ ] CPU Conv2d regression remains correct
- [ ] full regression suite passes
- [ ] architecture docs and educational example exist
- [ ] package version is `0.7.0`
- [ ] exact final v0.7 branch-head GitHub Actions is green

## 6. Recommended reading order

```text
mini_pytorch/device.py
mini_pytorch/backend.py
mini_pytorch/storage.py
mini_pytorch/tensor.py
mini_pytorch/dispatcher.py
mini_pytorch/operator_registry.py
tests/test_device_backend.py
docs/DEVICE_BACKEND.md
examples/10_device_backend.py
```

## 7. Next version

After v0.7 freezes, v0.8 should deepen Autograd runtime semantics: graph lifetime, retain_grad/hooks, retain_graph, and higher-order-gradient concepts.