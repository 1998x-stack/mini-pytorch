# mini-pytorch v0.1

## 1. Version identity

- **Branch:** `v0.1`
- **Historical base commit:** `00b544a85c0c48d92a47b00efdc299678244e6c3`
- **Theme:** minimal PyTorch learning MVP
- **Status:** frozen historical learning snapshot

v0.1 establishes the first complete, runnable mini-pytorch architecture. Its purpose is not API completeness; its purpose is to make the basic PyTorch training loop understandable from source code.

```text
Tensor
  ↓
Operator
  ↓
Dynamic computation graph
  ↓
Autograd / backward
  ↓
Parameter
  ↓
Module
  ↓
Loss
  ↓
Optimizer
  ↓
Training loop
```

---

## 2. What this version teaches

After reading v0.1, the learner should be able to explain what happens internally when executing:

```python
pred = model(x)
loss = loss_fn(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The key mental model is:

> A Tensor is data plus autograd metadata, and every differentiable operation can create a node in a dynamic computation graph.

---

## 3. Core functionality

### Tensor and Autograd

- `Tensor`
- `requires_grad`
- `grad`
- `grad_fn`
- leaf tensor semantics
- dynamic computation graph
- reverse-mode automatic differentiation
- topological traversal during `backward()`
- gradient accumulation
- `detach()`
- `no_grad()`
- scalar and explicit-gradient backward entry points

### Differentiable operations

Representative operators include:

- addition / subtraction
- multiplication / division
- power
- matrix multiplication
- sum / mean
- reshape / transpose
- ReLU
- sigmoid
- exp / log
- broadcasting-aware gradient reduction

### Neural-network abstraction

- `Parameter`
- `Module`
- recursive parameter registration
- `parameters()`
- `zero_grad()`
- `train()` / `eval()`
- `state_dict()` / `load_state_dict()`
- `Linear`
- `ReLU`
- `Sigmoid`
- `Sequential`
- `MSELoss`

### Optimizers

- extensible `Optimizer` base class
- parameter groups
- per-parameter optimizer state
- SGD
  - learning rate
  - momentum
  - weight decay
- Adam

### Utilities and learning aids

- numerical `gradcheck`
- computation-graph DOT export
- small regression / MLP examples
- unit tests around Tensor, Autograd, Module and Optimizer behavior

---

## 4. Difference from the previous repository state

There is no earlier numbered version. v0.1 is the **baseline version**.

Compared with the repository initialization stage, v0.1 introduces the complete first learning stack:

```text
repository skeleton
      ↓
Tensor + operators
      ↓
Autograd engine
      ↓
nn.Module / Parameter
      ↓
Linear / activations / loss
      ↓
Optimizer
      ↓
end-to-end trainable examples
```

This version should therefore be treated as the reference point for all later version comparisons.

---

## 5. Important source files

Recommended reading order:

```text
mini_pytorch/autograd.py
        ↓
mini_pytorch/tensor.py
        ↓
mini_pytorch/ops.py
        ↓
mini_pytorch/nn/module.py
        ↓
mini_pytorch/nn/linear.py
        ↓
mini_pytorch/optim/optimizer.py
        ↓
mini_pytorch/optim/sgd.py
        ↓
mini_pytorch/optim/adam.py
```

The intended learning sequence is dependency-first: understand Tensor/Autograd before reading Module and Optimizer abstractions.

---

## 6. What v0.1 intentionally does not implement

The following are intentionally out of scope:

- Dataset / DataLoader / Sampler
- MNIST data pipeline
- classification `CrossEntropyLoss`
- rich optimizer suite
- learning-rate schedulers
- explicit Storage abstraction
- explicit stride / storage offset semantics
- `view()` vs `reshape()` memory behavior
- `contiguous()` semantics
- Conv2d / pooling
- CUDA / devices
- distributed training
- multiprocessing DataLoader
- advanced autograd graph-lifetime controls

These omissions are deliberate: v0.1 isolates the smallest coherent deep-learning framework core.

---

## 7. Version boundary

v0.1 answers:

> How can a small NumPy-based framework reproduce the essential `Tensor → Autograd → Module → Optimizer` architecture of PyTorch?

It does **not** yet answer:

> How should optimizers evolve over time, how should learning rates be scheduled, or how should data be loaded and sampled?

Those questions begin in v0.2 and v0.3.

---

## 8. Next version

`v0.2` keeps the Tensor/Autograd/Module layers stable and expands the **optimization subsystem**:

```text
v0.1
basic SGD + Adam
      ↓
v0.2
optimizer family
parameter-group behavior
checkpointable optimizer state
learning-rate schedulers
```

For the exact delta, read `version.md` on the `v0.2` branch.
