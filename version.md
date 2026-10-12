# mini-pytorch v0.6

## 1. Version identity

- **Branch:** `v0.6`
- **Previous version:** `v0.5`
- **Theme:** neural-network operator stack / CNN
- **Status:** active development
- **Target package version:** `0.6.0`

v0.6 is the first release that uses the v0.5 Dispatcher for a genuinely non-trivial neural-network operator family.

```text
nn.Conv2d
    ↓
functional.conv2d
    ↓
aten::conv2d.default
    ↓
Dispatcher
    ├── Autograd wrapper
    │       ↓ redispatch
    └── CPU kernel
            ↓
       NumPy reference
```

The central question is:

> Can the operator/dispatcher architecture introduced in v0.5 support realistic trainable NN operators without pushing backend/autograd policy back into `Tensor`?

## 2. Delta from v0.5

v0.5 established execution routing:

```text
logical operator → schema → DispatchKeySet → kernel
```

v0.6 keeps that layer stable and builds a higher-level NN stack on top:

```text
Module
  ↓
functional API
  ↓
logical operator
  ↓
Dispatcher
  ↓
Autograd wrapper
  ↓
CPU numerical kernel
```

## 3. Required mechanisms

### Conv2d

- `nn.Conv2d`
- `functional.conv2d`
- `aten::conv2d.default`
- readable loop CPU kernel
- im2col CPU kernel
- reference-vs-im2col equivalence tests
- NCHW input
- weight shape `(out_channels, in_channels, kH, kW)`
- bias
- stride / padding / dilation
- `groups=1` in v0.6; unsupported groups fail clearly
- input / weight / bias gradients
- deterministic shape validation

### MaxPool2d

- `nn.MaxPool2d`
- `functional.max_pool2d`
- `aten::max_pool2d.default`
- forward max selection
- backward routing to winning input positions
- kernel size / stride / padding

### Dropout

- `nn.Dropout`
- train/eval semantics
- inverted-dropout scaling `1/(1-p)`
- deterministic seeded educational path

### BatchNorm

- `BatchNorm1d`
- `BatchNorm2d`
- affine `weight` / `bias` Parameters
- running mean / running variance Buffers
- train/eval behavior
- buffer state_dict coverage

### CNN MNIST

The end-to-end example reuses Dataset/DataLoader/Optimizer/Scheduler:

```text
N×1×28×28
   ↓
Conv2d → ReLU → MaxPool2d
   ↓
Conv2d → ReLU → MaxPool2d
   ↓
Flatten → Linear → CrossEntropyLoss
```

## 4. Acceptance criteria

v0.6 is complete when:

- [ ] Conv2d reference forward matches an independent NumPy reference
- [ ] Conv2d im2col output matches reference loop output
- [ ] Conv2d input/weight/bias gradients pass finite-difference checks
- [ ] Dispatcher routes Conv2d through Autograd → CPU
- [ ] MaxPool2d forward/backward are correct
- [ ] Dropout train/eval behavior is correct
- [ ] BatchNorm affine parameters and running-stat buffers are distinct
- [ ] BatchNorm train/eval behavior is correct
- [ ] CNN forward/backward/training step works end to end
- [ ] all previous tests remain green
- [ ] `docs/CNN_OPERATORS.md` explains the mechanism
- [ ] `examples/09_cnn_mnist.py` demonstrates the integrated stack
- [ ] exact final `v0.6` branch-head GitHub Actions run is green

## 5. Explicit non-goals

This version does **not** add:

- real CUDA or `.to(device)`
- grouped/depthwise convolution beyond a clear `groups=1` limitation
- transposed convolution
- adaptive pooling
- mixed precision
- fused BatchNorm/Conv kernels
- multiprocessing DataLoader
- distributed training
- compiler graph capture

## 6. Recommended reading order

```text
mini_pytorch/kernels.py
        ↓
mini_pytorch/operator_registry.py
        ↓
mini_pytorch/functional.py
        ↓
mini_pytorch/nn/conv.py
mini_pytorch/nn/pooling.py
mini_pytorch/nn/dropout.py
mini_pytorch/nn/batchnorm.py
        ↓
tests/test_cnn_ops.py
        ↓
docs/CNN_OPERATORS.md
        ↓
examples/09_cnn_mnist.py
```

## 7. Learning goals

By the end of v0.6, the reader should be able to explain:

1. why `nn.Conv2d` is mostly parameter ownership and configuration;
2. why `functional.conv2d` should call a logical operator rather than contain backend loops;
3. how Autograd can wrap a backend Conv2d kernel through redispatch;
4. how convolution backward accumulates gradients for input, weight and bias;
5. why im2col turns convolution into matrix multiplication;
6. why pooling backward needs winner information;
7. why Dropout changes behavior between train/eval;
8. why BatchNorm running statistics are buffers rather than Parameters.

## 8. Next version

`v0.7` should make backend identity explicit with `Device`, device metadata on Tensor/Storage, backend-owned allocation and device-aware dispatch.