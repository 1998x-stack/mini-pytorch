# Agents.md — mini-pytorch v0.6 Development Contract

> Read this file before modifying the `v0.6` branch.

## Branch identity

- Branch: `v0.6`
- Previous version: `v0.5`
- Primary mental model: **neural-network operator stack / CNN built on Dispatcher**
- Status: active until the v0.6 Definition of Done is satisfied, then frozen

## 1. Scope rule

v0.6 teaches one architectural transition:

```text
Module API
   ↓
functional operator
   ↓
Dispatcher
   ↓
Autograd wrapper
   ↓ redispatch
CPU numerical kernel
```

The primary scope is:

- Conv2d
- MaxPool2d
- Dropout
- BatchNorm1d / BatchNorm2d
- CNN MNIST integration

Do not turn v0.6 into a general device/backend release.

## 2. Mandatory reading order

Before code changes, read:

1. `Agents.md`
2. `version.md`
3. `docs/DISPATCHER.md`
4. `mini_pytorch/dispatcher.py`
5. `mini_pytorch/operator_registry.py`
6. `mini_pytorch/kernels.py`
7. relevant `nn/*`, tests and examples

## 3. Conv2d architecture contract

Required call path:

```text
nn.Conv2d.forward
      ↓
functional.conv2d
      ↓
aten::conv2d.default
      ↓
Dispatcher
      ↓
Autograd wrapper when needed
      ↓ redispatch
CPU Conv2d kernel
```

Forbidden shortcuts:

- putting convolution loops directly in `nn.Conv2d.forward`;
- bypassing Dispatcher for the main Conv2d implementation;
- embedding Autograd graph policy in the CPU kernel;
- adding device-specific branches in `Tensor`.

## 4. Reference-first performance rule

Implementation order:

```text
readable nested-loop Conv2d
        ↓
correctness + gradient tests
        ↓
im2col implementation
        ↓
reference/im2col equivalence
        ↓
optional benchmark
```

Keep both paths. The readable reference implementation is part of the learning artifact.

## 5. Supported Conv2d surface

v0.6 should support:

- NCHW input
- OIHW weights
- optional bias represented through the functional/module API
- integer or pair `kernel_size`
- integer or pair `stride`
- integer or pair `padding`
- integer or pair `dilation`
- `groups=1`

If `groups != 1`, raise a clear error. Do not silently pretend grouped convolution is supported.

## 6. Pooling contract

MaxPool2d should:

- route through Dispatcher;
- preserve NCHW shape reasoning;
- define deterministic tie behavior through NumPy first-max semantics;
- route backward gradient only to selected maxima;
- expose kernel size / stride / padding.

## 7. Dropout contract

Dropout should demonstrate policy rather than backend complexity:

- training randomly masks elements;
- surviving activations scale by `1/(1-p)`;
- eval is identity;
- deterministic tests must be possible with a seed;
- no generated masks/checkpoints are committed.

## 8. BatchNorm contract

BatchNorm must make Parameter vs Buffer visible:

```text
weight / bias        → Parameters
running_mean / var   → Buffers
```

Requirements:

- training uses batch statistics;
- evaluation uses running statistics when tracking is enabled;
- affine parameters are optional;
- running statistics participate in `state_dict()` but not `parameters()`;
- running-stat updates are tracked framework mutations.

## 9. Testing contract

Before freeze, tests must cover:

- Conv2d forward reference values;
- Conv2d shape validation;
- Conv2d input/weight/bias finite-difference gradients;
- Dispatcher route `Autograd → CPU`;
- im2col/reference equivalence;
- MaxPool2d forward/backward;
- Dropout train/eval and deterministic seed behavior;
- BatchNorm train/eval, parameter/buffer distinction, state_dict;
- one CNN end-to-end forward/backward/optimizer step;
- complete regression suite.

Full command:

```bash
python -m unittest discover -s tests -v
```

Exact final v0.6 branch-head GitHub Actions must be green before freeze.

## 10. Documentation contract

v0.6 must include/update:

- `README.md`
- `version.md`
- `Agents.md`
- `docs/CNN_OPERATORS.md`
- `examples/09_cnn_mnist.py`
- package version

## 11. Explicit non-goals

Do not add in v0.6:

- CUDA execution
- `.to(device)`
- distributed training
- graph compiler work
- ConvTranspose
- full grouped/depthwise convolution
- mixed precision
- production-grade kernel selection/autotuning

## 12. Freeze rule

When all v0.6 acceptance criteria pass:

1. change `version.md` status to frozen;
2. make README identify a frozen v0.6 snapshot;
3. run CI on the exact final branch head;
4. integrate the validated v0.6 tree into `main`;
5. restore canonical `main/Agents.md` and project README;
6. advance `main/Plan.md` to v0.7 Device / Backend.