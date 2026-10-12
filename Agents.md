# Agents.md — mini-pytorch v0.7 Frozen Branch Contract

> Read this file before modifying the `v0.7` branch.

## Branch identity

- Branch: `v0.7`
- Previous version: `v0.6`
- Primary mental model: **Device + Backend + device-aware Dispatcher**
- Status: **frozen historical learning snapshot**

## 1. What v0.7 teaches

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
   ├── Autograd / NoGrad policy
   ├── CPU backend
   └── EDU simulated backend
```

The version scope is intentionally limited to:

- Device value object;
- backend registry/ownership;
- Storage.device / Tensor.device;
- Tensor.to(device);
- Module.to(device);
- device-aware DispatchKey inference;
- mixed-device validation;
- CPU + deterministic EDU backend routing.

## 2. Historical branch rule

This branch is frozen.

Allowed changes:

- documentation corrections;
- tests clarifying promised v0.7 behavior;
- narrowly scoped correctness fixes within Device/Backend semantics.

Forbidden:

- v0.8 advanced Autograd features;
- DDP;
- real CUDA/CuPy runtime;
- streams/events;
- compiler features;
- unrelated API expansion.

## 3. Frozen semantics

### Transfer

```text
x.to(x.device, copy=False) → x
same-device copy=True      → new Storage
cross-device transfer      → new target Storage
```

Cross-device transfer breaks Storage aliasing and preserves gradient connectivity through identity CopyBackward when grad mode is enabled.

### Dispatcher

```text
CPU + grad  → {Autograd, CPU}
EDU + grad  → {Autograd, EDU}
CPU no_grad → {NoGrad, CPU}
EDU no_grad → {NoGrad, EDU}
```

Mixed-device Tensor operands fail before backend numerical execution.

### Simulated EDU backend

`edu` uses NumPy host arrays internally and exists only to demonstrate backend identity/routing without hardware dependency. Never describe it as CUDA or GPU acceleration.

## 4. Validation baseline

Release-candidate validation:

```text
103 tests
OK
```

Before reporting any future v0.7 fix as validated, rerun the full suite and require GitHub Actions success on the exact modified branch head.

## 5. Documentation map

```text
README.md
version.md
docs/DEVICE_BACKEND.md
examples/10_device_backend.py
tests/test_device_backend.py
```

## 6. Next-version boundary

Advanced Autograd Engine work belongs to `v0.8`, created from the latest validated `main`, not from this frozen branch.