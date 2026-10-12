# Agents.md — mini-pytorch v0.6 Frozen Branch Contract

> Read this file before modifying the `v0.6` branch.

## Branch identity

- Branch: `v0.6`
- Previous version: `v0.5`
- Primary mental model: **neural-network operator stack / CNN built on Dispatcher**
- Status: **frozen historical learning snapshot**

## 1. What v0.6 teaches

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

The version scope is intentionally limited to:

- Conv2d
- MaxPool2d
- Dropout
- BatchNorm1d / BatchNorm2d
- CNN MNIST integration

## 2. Historical branch rule

This branch is frozen.

Allowed changes:

- documentation corrections;
- tests that clarify v0.6 promised behavior;
- narrowly scoped bug fixes inside the v0.6 feature boundary.

Forbidden changes:

- v0.7 Device/backend work;
- CUDA execution;
- `.to(device)`;
- distributed training;
- compiler work;
- future-version APIs merely for convenience.

Future development must begin from latest `main`, not from this frozen branch.

## 3. Required reading order

To study v0.6:

1. `version.md`
2. `README.md`
3. `docs/DISPATCHER.md`
4. `docs/CNN_OPERATORS.md`
5. `mini_pytorch/kernels.py`
6. `mini_pytorch/nn_ops.py`
7. `mini_pytorch/operator_registry.py`
8. `mini_pytorch/functional.py`
9. `mini_pytorch/nn/conv.py`
10. `mini_pytorch/nn/pooling.py`
11. `mini_pytorch/nn/dropout.py`
12. `mini_pytorch/nn/batchnorm.py`
13. `tests/test_cnn_ops.py`
14. `examples/09_cnn_mnist.py`

## 4. Conv2d architecture invariant

The intended call path is:

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

Do not move convolution loops into `nn.Conv2d.forward`.
Do not duplicate Autograd policy inside the CPU kernel.

## 5. Reference-first kernel invariant

v0.6 deliberately keeps both:

```text
readable nested-loop reference
            +
im2col + matrix multiplication
```

The optimized educational path must remain numerically equivalent to the readable reference path.

## 6. Supported Conv2d boundary

v0.6 supports:

- NCHW input;
- OIHW weights;
- bias;
- stride;
- padding;
- dilation;
- `groups=1`.

`groups != 1` must fail clearly. Grouped/depthwise convolution belongs to a later expansion if explicitly planned.

## 7. State semantics

BatchNorm preserves the v0.6 state distinction:

```text
weight / bias
    ↓
Parameters

running_mean / running_var / num_batches_tracked
    ↓
Buffers
```

Buffers belong in `state_dict()` but not `parameters()`.

## 8. Validation contract

The release-candidate implementation passed:

```text
90 tests
OK
```

Coverage includes:

- Conv2d reference values;
- im2col/reference equivalence;
- finite-difference input/weight/bias gradients;
- Dispatcher Autograd route;
- MaxPool2d forward/backward;
- Dropout train/eval/seed behavior;
- BatchNorm state and train/eval behavior;
- CNN end-to-end optimizer step;
- all previous regression tests.

Any future permitted bug fix on this branch must run the complete suite:

```bash
python -m unittest discover -s tests -v
```

## 9. Explicit non-goals

Do not add here:

- device abstraction;
- accelerator allocation;
- CUDA kernels;
- ConvTranspose;
- full grouped/depthwise convolution;
- distributed training;
- graph capture/compiler subsystems.

## 10. Next development location

The next planned version is `v0.7` and must be created from latest `main` after v0.6 integration.

v0.7's mental model is **Device / Backend abstraction**, not additional CNN feature accumulation.