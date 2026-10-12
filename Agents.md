# Agents.md — mini-pytorch v0.7 Development Contract

> Read this file before modifying the `v0.7` branch.

## Branch identity

- Branch: `v0.7`
- Previous version: `v0.6`
- Mental model: **Device + Backend + device-aware Dispatcher**
- Status: active until Definition of Done is satisfied, then freeze.

## 1. Scope

v0.7 exists to make placement explicit without real accelerator dependencies.

Allowed scope:

- Device value object
- backend registry/ownership
- CPU backend
- deterministic simulated `edu` backend
- Storage.device / Tensor.device
- Tensor.to(device)
- device-aware DispatchKey inference
- cross-device validation
- preserving Autograd above backend routing

Do not add advanced Autograd, DDP, compiler graph capture, streams, CUDA kernels, or mixed precision.

## 2. Architecture invariant

Preferred direction:

```text
Tensor metadata
   ↓
Storage + Device
   ↓
Backend-owned allocation
   ↓
logical operator
   ↓
Dispatcher
   ├── Autograd wrapper
   ├── CPU backend key
   └── EDU backend key
```

Forbidden pattern:

```python
if tensor.device == "cpu":
    ...
elif tensor.device == "edu":
    ...
```

repeated across Tensor/operator implementations.

Backend selection belongs in the Dispatcher/backend registry.

## 3. Simulated backend rule

`edu` may use NumPy arrays internally. Its purpose is to validate architecture, not simulate performance.

It must nevertheless have:

- distinct `Device` identity;
- distinct DispatchKey;
- independent backend registration;
- transfer semantics through `.to()`;
- clear mixed-device errors.

Do not claim `edu` is GPU or CUDA.

## 4. Transfer semantics

Target semantics:

- `x.to(x.device)` returns `x` when `copy=False`;
- `copy=True` creates new Storage;
- changing device creates new Storage and breaks aliasing;
- dtype is preserved unless a future version explicitly adds dtype conversion;
- requires-grad is preserved;
- cross-device copy remains connected to Autograd with identity gradient;
- views transfer their logical values, not their original Storage alias graph.

## 5. Dispatcher semantics

Mixed devices among Tensor operands must fail before numerical backend execution.

Expected routes:

```text
CPU + grad    → {Autograd, CPU}
EDU + grad    → {Autograd, EDU}
CPU no-grad   → {NoGrad, CPU}
EDU no-grad   → {NoGrad, EDU}
```

Autograd wrappers must redispatch after removing only the policy key, preserving the backend key.

## 6. Testing contract

Add tests for:

- Device parsing/equality/hash/errors;
- backend registry and allocation;
- Storage/Tensor device identity;
- `.to()` same-device/copy/cross-device behavior;
- transfer gradient flow;
- alias separation after transfer;
- Dispatcher CPU vs EDU selection;
- mixed-device errors;
- scalar promotion on EDU;
- direct Function.apply operator device preservation;
- Conv2d/MaxPool routing on EDU where supported;
- full previous 90-test regression.

Before freeze, exact final branch-head GitHub Actions must be green.

## 7. Documentation contract

Before freeze update:

- `version.md`
- `README.md`
- `Agents.md`
- `docs/DEVICE_BACKEND.md`
- `examples/10_device_backend.py`

## 8. Freeze rule

Once frozen, v0.7 becomes a historical learning snapshot. No v0.8 features may be backported except narrow correctness fixes inside the v0.7 scope.

Validated v0.7 must then be integrated into `main`, and main governance docs must be restored/advanced to v0.8.