# mini-pytorch v0.6

## 1. Version identity

- **Branch:** `v0.6`
- **Previous version:** `v0.5`
- **Theme:** neural-network operator stack / CNN
- **Status:** **frozen historical learning snapshot**
- **Package version:** `0.6.0`
- **Validated regression baseline:** **90/90 tests passed**

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

The central question answered by this version is:

> Can the operator/dispatcher architecture introduced in v0.5 support realistic trainable NN operators without pushing backend/autograd policy back into `Tensor`?

## 2. Core difference from v0.5

v0.5 established execution routing:

```text
logical operator → schema → DispatchKeySet → kernel
```

v0.6 builds a higher-level NN stack on top:

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

## 3. Implemented mechanisms

### Conv2d

Implemented:

- `nn.Conv2d`
- `functional.conv2d`
- `aten::conv2d.default`
- readable nested-loop CPU kernel
- im2col + matrix-multiplication CPU kernel
- reference-vs-im2col equivalence tests
- NCHW input
- OIHW weights
- bias
- stride
- padding
- dilation
- explicit `groups=1` boundary
- input / weight / bias backward
- shape and error validation

Autograd route:

```text
DispatchKeySet{Autograd, CPU}
      ↓
Autograd wrapper
      ↓ save input/weight/config
redispatch {CPU}
      ↓
CPU Conv2d
      ↓
attach Conv2dOpBackward
```

### MaxPool2d

Implemented:

- `nn.MaxPool2d`
- `functional.max_pool2d`
- `aten::max_pool2d.default`
- Dispatcher CPU + Autograd registration
- kernel size / stride / padding
- backward routing to selected maxima

### Dropout

Implemented:

- `nn.Dropout`
- functional dropout
- inverted-dropout scaling `1/(1-p)`
- train/eval behavior
- deterministic seeded tests

Dropout is intentionally composed from an explicit mask and the existing Tensor multiply path instead of inventing a backend kernel with no new backend mechanism to teach.

### BatchNorm

Implemented:

- `BatchNorm1d`
- `BatchNorm2d`
- affine `weight` / `bias` Parameters
- `running_mean` / `running_var` Buffers
- `num_batches_tracked` Buffer
- train/eval behavior
- state_dict buffer coverage
- biased batch variance for normalization
- unbiased running-variance observation when sample count permits

### CNN MNIST

Added `examples/09_cnn_mnist.py`:

```text
MNIST
  ↓
DataLoader
  ↓
N×1×28×28
  ↓
Conv2d → BatchNorm2d → ReLU → MaxPool2d
  ↓
Conv2d → ReLU → MaxPool2d
  ↓
Flatten → Dropout → Linear
  ↓
CrossEntropyLoss
  ↓
AdamW + CosineAnnealingLR
```

The example preserves Loguru logging, CSV metrics, loss/accuracy plots and checkpoints from the earlier MNIST training stack.

## 4. New and modified source files

New:

```text
mini_pytorch/nn_ops.py
mini_pytorch/nn/conv.py
mini_pytorch/nn/pooling.py
mini_pytorch/nn/dropout.py
mini_pytorch/nn/batchnorm.py
tests/test_cnn_ops.py
docs/CNN_OPERATORS.md
examples/09_cnn_mnist.py
```

Major modifications:

```text
mini_pytorch/kernels.py
mini_pytorch/operator_registry.py
mini_pytorch/functional.py
mini_pytorch/nn/__init__.py
mini_pytorch/__init__.py
pyproject.toml
README.md
```

## 5. Validation

Core implementation release candidate:

```text
75173ec4790e9f32e8edc0b59d4344f1000c5400
```

GitHub Actions environment:

```text
Ubuntu 24.04
Python 3.11.17
NumPy 2.4.6
```

Result:

```text
Ran 90 tests in 0.812s
OK
```

New v0.6 coverage includes:

- known-value Conv2d forward
- loop vs im2col equivalence
- finite-difference Conv2d input gradients
- finite-difference Conv2d weight gradients
- finite-difference Conv2d bias gradients
- Dispatcher Autograd selection for Conv2d
- unsupported groups error
- MaxPool2d forward/backward
- Dropout train/eval/seed behavior
- BatchNorm Parameter/Buffer distinction
- BatchNorm train/eval running-stat behavior
- end-to-end CNN backward + optimizer step

## 6. Definition of Done

- [x] Conv2d reference forward is validated
- [x] Conv2d im2col matches reference loops
- [x] Conv2d input/weight/bias gradients pass finite differences
- [x] Dispatcher routes Conv2d through Autograd → CPU
- [x] MaxPool2d forward/backward are correct
- [x] Dropout train/eval behavior is tested
- [x] BatchNorm Parameters and Buffers are distinct
- [x] BatchNorm train/eval behavior is tested
- [x] CNN forward/backward/optimizer step works end to end
- [x] full previous regression remains green
- [x] `docs/CNN_OPERATORS.md` explains the architecture
- [x] `examples/09_cnn_mnist.py` provides an end-to-end training path
- [x] package version is `0.6.0`
- [x] branch is ready for exact final-head CI and integration into `main`

## 7. Recommended reading order

```text
mini_pytorch/kernels.py
        ↓
mini_pytorch/nn_ops.py
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

## 8. Explicit non-goals

v0.6 intentionally does not implement:

- real CUDA or `.to(device)`
- grouped/depthwise Conv2d beyond the explicit `groups=1` boundary
- transposed convolution
- adaptive pooling
- mixed precision
- fused Conv/BatchNorm kernels
- multiprocessing DataLoader
- distributed training
- compiler graph capture

## 9. Learning goals

After v0.6, the reader should be able to explain:

1. why `nn.Conv2d` owns state but not backend loops;
2. why functional operators are a useful boundary;
3. how Dispatcher-backed Autograd redispatches to a CPU Conv2d kernel;
4. how convolution backward produces dInput/dWeight/dBias;
5. how im2col lowers sliding windows to GEMM;
6. why pooling backward needs winner positions;
7. why Dropout has different train/eval behavior;
8. why BatchNorm running statistics are Buffers rather than Parameters.

## 10. Next version

`v0.7` should make backend identity explicit:

```text
Tensor / Storage
      ↓
Device metadata
      ↓
Dispatcher
   ┌────┴────┐
   ▼         ▼
 CPU       optional accelerator backend
```

The next major question is:

> How can one Tensor/operator API support multiple devices without scattering device-specific conditionals throughout the framework?