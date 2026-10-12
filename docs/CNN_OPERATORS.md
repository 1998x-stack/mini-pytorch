# CNN Operators in mini-pytorch v0.6

v0.6 answers one architectural question:

> How does a high-level neural-network module become a backend kernel call while keeping Autograd and backend math separated?

The representative path is Conv2d.

```text
nn.Conv2d
    ↓ owns Parameters/config
functional.conv2d
    ↓ calls logical operator
aten::conv2d.default
    ↓
Dispatcher
    ↓
DispatchKeySet
 ┌─────────────┐
 │             │
Autograd      CPU
 │             │
 └─redispatch─►│
               ↓
        NumPy Conv2d kernel
```

## 1. Module vs functional API vs kernel

`nn.Conv2d` does **not** contain convolution loops.

Its responsibilities are:

- own `weight` and optional `bias` Parameters;
- remember stride/padding/dilation configuration;
- initialize parameters;
- call `functional.conv2d()`.

`functional.conv2d()` does not own the numerical implementation either. It normalizes arguments and calls:

```text
aten::conv2d.default
```

The Dispatcher then chooses an implementation.

This separation mirrors the conceptual boundary in real PyTorch:

```text
Module state
   !=
logical operator
   !=
backend kernel
```

## 2. Conv2d is cross-correlation

For NCHW input and OIHW weights:

```text
input:  (N, C_in, H_in, W_in)
weight: (C_out, C_in, K_h, K_w)
bias:   (C_out,)
```

The output is:

```text
(N, C_out, H_out, W_out)
```

with

```text
H_out = floor((H_in + 2P_h - D_h(K_h-1) - 1) / S_h) + 1
W_out = floor((W_in + 2P_w - D_w(K_w-1) - 1) / S_w) + 1
```

The implementation performs cross-correlation, matching the convention used by deep-learning frameworks: the kernel is not spatially flipped before the forward calculation.

v0.6 supports:

- NCHW input;
- OIHW weights;
- bias;
- stride;
- zero padding;
- dilation;
- `groups=1`.

Grouped/depthwise convolution is deliberately deferred.

## 3. Readable reference kernel

`kernels.conv2d_reference()` uses explicit loops:

```text
batch
  ↓
out_channel
  ↓
out_h
  ↓
out_w
  ↓
in_channel
  ↓
kernel_h
  ↓
kernel_w
```

This is slow, but every index is visible.

For one output element:

```text
out[n, co, ho, wo]
    = bias[co]
    + Σ input_padded[n, ci, ho*S_h + kh*D_h, wo*S_w + kw*D_w]
        * weight[co, ci, kh, kw]
```

The reference path is intentionally kept even after an optimized educational path is added.

## 4. im2col turns convolution into matrix multiplication

The second implementation is:

```text
input windows
    ↓
im2col
    ↓
2-D matrix
    ↓
GEMM / matmul
    ↓
reshape + transpose
    ↓
NCHW output
```

Conceptually:

```text
columns.shape = (N * H_out * W_out, C_in * K_h * K_w)
weight_matrix.shape = (C_out, C_in * K_h * K_w)

output_matrix = columns @ weight_matrix.T
```

Then:

```text
(N*H_out*W_out, C_out)
        ↓
(N, H_out, W_out, C_out)
        ↓ transpose
(N, C_out, H_out, W_out)
```

Tests require the im2col and nested-loop paths to produce the same numerical result.

## 5. Autograd wrapper and redispatch

When any Conv2d Tensor requires gradients:

```text
DispatchKeySet = {Autograd, CPU}
```

The Dispatcher selects `Autograd` first.

The Autograd wrapper:

1. saves input/weight and convolution configuration in a Context;
2. removes the `Autograd` key;
3. redispatches to `{CPU}`;
4. receives the numerical output;
5. attaches `Conv2dOpBackward`.

The CPU kernel therefore has no `requires_grad` logic.

## 6. Conv2d backward

Given upstream gradient `dY`, backward computes three values.

### Bias gradient

Every output location using the same output channel contributes to its bias:

```text
dB[co] = Σ dY[n, co, ho, wo]
```

### Weight gradient

Each weight accumulates input values multiplied by the corresponding upstream gradient:

```text
dW[co, ci, kh, kw]
    += dY[n, co, ho, wo]
       * X_padded[n, ci, ho*S_h + kh*D_h, wo*S_w + kw*D_w]
```

### Input gradient

Each input location receives contributions from every output element whose receptive field included it:

```text
dX_padded[n, ci, ih, iw]
    += dY[n, co, ho, wo] * W[co, ci, kh, kw]
```

After accumulation, padding is cropped away.

v0.6 validates these gradients with finite differences on a small convolution.

## 7. MaxPool2d

The forward rule is:

```text
window → maximum
```

Backward is not an average. The incoming gradient is routed to the winning input position:

```text
2×2 input window
[1  8]
[3  4]

max = 8

upstream grad = g

input grad
[0  g]
[0  0]
```

The educational implementation recomputes the winner during backward using deterministic NumPy first-maximum behavior.

## 8. Dropout

During training, each element is independently kept with probability `1-p`.

v0.6 uses inverted dropout:

```text
output = input * mask / (1-p)
```

Therefore:

```text
train: random mask + scaling
eval:  identity
```

Dropout is built from an explicit non-grad mask Tensor and the existing dispatched multiplication operator. It does not need a new backend kernel merely to demonstrate the mechanism.

A seed is available for deterministic tests and examples.

## 9. BatchNorm: Parameters vs Buffers

For C channels:

```text
weight (gamma) → Parameter
bias   (beta)  → Parameter

running_mean → Buffer
running_var  → Buffer
num_batches_tracked → Buffer
```

This distinction matters:

- Parameters are returned by `model.parameters()` and optimized;
- Buffers appear in `state_dict()` but are not optimized.

Training normalization uses current batch statistics:

```text
mean = E[x]
var  = E[(x-mean)^2]
y = (x-mean) / sqrt(var + eps)
y = y * gamma + beta
```

Running statistics are updated for evaluation.

The normalization variance used during the training forward is the biased batch variance. The running variance update uses the unbiased sample estimate when more than one sample contributes to the channel statistic.

Evaluation with tracked statistics uses:

```text
running_mean
running_var
```

instead of the current batch's statistics.

## 10. CNN MNIST stack

The v0.6 integration example is:

```text
MNIST Dataset
   ↓
DataLoader
   ↓
reshape N×28×28 → N×1×28×28
   ↓
Conv2d(1 → 8)
   ↓
BatchNorm2d
   ↓
ReLU
   ↓
MaxPool2d
   ↓
Conv2d(8 → 16)
   ↓
ReLU
   ↓
MaxPool2d
   ↓
Flatten
   ↓
Dropout
   ↓
Linear
   ↓
CrossEntropyLoss
   ↓
AdamW + LR Scheduler
```

This reuses subsystems from earlier versions instead of introducing a separate training framework.

## 11. Reading order

Recommended source walk:

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
examples/09_cnn_mnist.py
```

## 12. What v0.6 intentionally does not solve

- CUDA/device allocation;
- backend-owned Storage allocation;
- grouped/depthwise convolution;
- ConvTranspose;
- kernel autotuning;
- mixed precision;
- fused convolution/normalization;
- distributed training;
- graph compilation.

Those omissions preserve the version boundary. v0.7 is where device/backend identity becomes explicit.