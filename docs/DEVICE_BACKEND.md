# Device / Backend Architecture — mini-pytorch v0.7

v0.7 adds an explicit answer to a question that earlier versions deliberately avoided:

> Where does a Tensor live, who owns its allocation, and how does an operator choose the matching backend implementation?

## 1. The problem

Without a placement abstraction, a framework tends to grow code like:

```python
if tensor.device == "cpu":
    ...
elif tensor.device == "cuda":
    ...
```

inside every Tensor method and every operator. That mixes three separate responsibilities:

1. **Tensor metadata** — shape, stride, dtype, device;
2. **allocation ownership** — which backend creates/interprets Storage;
3. **operator routing** — which registered implementation should run.

v0.7 separates them.

## 2. Architecture

```text
                       Tensor
                         │
        shape / stride / dtype / device
                         │
                         ▼
                       Storage
                         │
                 backend-owned array
                         │
                         ▼
                    logical operator
                         │
                         ▼
                     Dispatcher
              ┌──────────┴──────────┐
              ▼                     ▼
           CPU key                EDU key
              │                     │
              └──── backend kernel ─┘
                         ▲
                         │
                  Autograd wrapper
```

Autograd and NoGrad remain **policy keys above backend keys**.

Typical routes:

```text
CPU + requires_grad → {Autograd, CPU}
EDU + requires_grad → {Autograd, EDU}
CPU inside no_grad  → {NoGrad, CPU}
EDU inside no_grad  → {NoGrad, EDU}
```

The Autograd wrapper removes only `Autograd`, then redispatches while preserving the backend key.

## 3. Device

`mini_pytorch.device.Device` is an immutable value object.

```python
Device("cpu")
Device("edu")
Device("edu:0")
```

v0.7 supports `cpu` and `edu` only.

`edu` is **not CUDA**. It is a simulated backend identity that uses host NumPy memory so multi-backend routing can be tested on ordinary CI.

## 4. Backend registry

`mini_pytorch.backend` owns allocation policy.

```text
Backend
├── allocate(size, dtype)
├── from_numpy(array)
└── to_numpy(allocation)
```

Registered backends:

```text
cpu → NumPyBackend
edu → EducationalBackend
```

Both use NumPy arrays in v0.7, but they are distinct backend identities. This is intentional: architecture is introduced before hardware-specific optimization.

## 5. Storage ownership

Before v0.7:

```text
Storage ≈ flat NumPy array
```

Now:

```text
Storage
├── Device
├── Backend
├── backend-owned allocation
└── shared VersionCounter
```

Tensor views still share one Storage, therefore they also share one device.

A view cannot independently live on another device.

## 6. Tensor.to

```python
x.to("edu")
```

Semantics:

### same device, copy=False

```python
x.to(x.device) is x
```

No allocation occurs.

### same device, copy=True

A new Storage is allocated, values are copied, and aliasing is broken.

### different device

A new target-backend Storage is created from the Tensor's **logical values**.

This means transferring a non-contiguous view materializes it as a contiguous target Tensor.

```text
source view
   │
logical values
   ▼
new target Storage
```

The target does not alias source Storage.

## 7. Transfer and Autograd

When grad mode is enabled and the source requires gradients, a device copy is represented by an identity-gradient node:

```text
x (cpu, requires_grad)
   ↓
CopyToDevice
   ↓
y (edu)
   ↓
operator graph
   ↓
backward
   ↓
grad flows back to x
```

The educational backward is simply:

```text
dL/dx = dL/dy
```

because v0.7 copies values without dtype conversion.

Inside `no_grad()`, transfer does not create a graph.

## 8. Device-aware Dispatcher

The Dispatcher inspects all Tensor arguments before backend math.

If they are on multiple devices:

```text
cpu Tensor + edu Tensor
        ↓
DispatchError
```

There is no implicit cross-device arithmetic.

For one common device, Dispatcher maps:

```text
cpu → DispatchKey.CPU
edu → DispatchKey.EDU
```

The same logical operator is registered under both keys.

```text
aten::add.Tensor
├── Autograd wrapper
├── EDU backend registration
└── CPU backend registration
```

v0.7 intentionally lets CPU and EDU registrations reuse the same readable NumPy numerical function. The important point is that routing selects a backend registration before execution.

## 9. Scalar promotion

Expressions such as:

```python
x_edu + 3.0
2.0 * x_edu
```

must not silently construct a CPU scalar Tensor.

`Tensor._ensure_tensor()` creates scalar operands on the owning Tensor's device, preserving backend consistency.

## 10. Direct Function.apply operators

Not every operator had moved to the Dispatcher by v0.6. Operations such as `exp`, `log`, some layout ops, and division still use `Function.apply` directly.

v0.7 makes that path device-aware too:

- common input device is checked before forward;
- mixed-device inputs fail;
- output Tensor inherits the input device;
- genuine views continue sharing the same Storage/device.

This prevents device semantics from depending on whether an operator has already migrated to Dispatcher.

## 11. Module.to

`Module.to(device)` recursively moves:

- Parameters;
- Buffers;
- child modules.

Parameter objects are preserved rather than replaced so module registration remains stable.

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, 3),
    nn.BatchNorm2d(8),
)

model.to("edu")
```

Afterward both Conv2d parameters and BatchNorm running-stat buffers live on `edu`.

For this educational framework, call `model.to(...)` **before creating the optimizer**.

## 12. Why EDU exists

A real CUDA backend would require:

- accelerator allocation;
- CUDA kernels or an external library;
- hardware-aware CI;
- streams/synchronization;
- transfer semantics beyond a simple host copy.

Those concepts would obscure the v0.7 lesson.

`edu` allows the project to prove these invariants now:

```text
one Tensor API
      ↓
explicit Device
      ↓
backend-owned Storage
      ↓
Device-aware Dispatcher
      ↓
different backend registration
```

without pretending to provide GPU performance.

## 13. Deliberate non-goals

v0.7 does not implement:

- CUDA;
- CuPy;
- pinned memory;
- async copies;
- streams/events;
- mixed precision;
- allocator caching;
- distributed execution.

Those require additional runtime concepts and should be introduced only after the placement model is clear.

## 14. Recommended source reading

```text
mini_pytorch/device.py
        ↓
mini_pytorch/backend.py
        ↓
mini_pytorch/storage.py
        ↓
mini_pytorch/tensor.py
        ↓
mini_pytorch/dispatcher.py
        ↓
mini_pytorch/operator_registry.py
        ↓
mini_pytorch/nn/module.py
        ↓
tests/test_device_backend.py
        ↓
examples/10_device_backend.py
```
