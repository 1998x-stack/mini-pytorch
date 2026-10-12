# Advanced Autograd Engine — mini-pytorch v0.8

v0.8 changes the mental model from:

```text
Tensor.backward()
    ↓
DFS embedded inside Tensor
```

to:

```text
Tensor.backward()
    ↓
AutogradEngine.backward()
    ↓
explicit execution + lifetime policy
```

The goal is not API completeness. The goal is to make graph execution state visible.

---

## 1. Four kinds of Autograd state

A useful way to read the code is to separate four different states.

### 1.1 Tensor value state

```text
Tensor
├── Storage
├── shape / stride / offset
├── dtype
└── device
```

This survives backward.

### 1.2 Graph structure

```text
Tensor._grad_fn
     ↓
GradFn
├── Function class
├── parent Tensors
└── Context
```

The educational graph remains inspectable after backward, but its Context can become released.

### 1.3 Backward saved state

```text
Context
├── saved_tensors
├── input references used for version checks
├── saved Storage versions
└── released flag
```

This is what default backward releases.

### 1.4 Engine-transient gradient state

During one backward call:

```text
local_grads[id(tensor)] -> ndarray
```

This dictionary belongs to the Engine and disappears when the call finishes.

That is different from user-visible:

```text
Tensor.grad
```

---

## 2. Why graph lifetime exists

Suppose:

```python
x = Tensor(2.0, requires_grad=True)
loss = (x * x).sum()
loss.backward()
```

The multiply backward needs the original `x` values.

Forward saves those values in Context:

```text
Mul Context
└── saved_tensors = (x, x)
```

After backward, keeping every saved forward array forever would retain unnecessary memory.

v0.8 therefore does:

```text
successful backward
      ↓
release graph contexts
      ↓
saved_tensors = ()
input references = ()
saved-version metadata = ()
released = True
```

The Tensor values themselves still exist.

A second backward through the same released graph fails before hooks or `.grad` side effects occur.

---

## 3. `retain_graph` is not `retain_grad`

These names look similar but control different things.

### `retain_graph=True`

Controls **backward-state lifetime**:

```python
loss.backward(retain_graph=True)
loss.backward()
```

First execution preserves Context state.

Second execution can reuse it and, because it is not retained, releases it afterward.

### `tensor.retain_grad()`

Controls **whether a non-leaf Tensor stores its gradient persistently**.

```python
hidden = x * x
hidden.retain_grad()
loss = hidden.sum()
loss.backward()
```

Without `retain_grad()`:

```text
hidden.grad = None
```

With it:

```text
hidden.grad = dLoss/dHidden
```

### `zero_grad()`

Controls persistent accumulated gradients:

```text
retain_graph -> can the graph execute again?
retain_grad  -> should this non-leaf expose .grad?
zero_grad    -> clear a Tensor's persistent .grad value
```

---

## 4. Engine execution order

For an output Tensor:

```text
output
  ↓
build topological order
  ↓
preflight: all reachable Contexts available?
  ↓
seed output gradient
  ↓
reverse topological traversal
```

At each Tensor node:

```text
incoming local gradient
        ↓
run Tensor hooks
        ↓
possibly transformed gradient
        ↓
leaf or retain_grad? ──yes──> accumulate Tensor.grad
        ↓
GradFn exists?
        ↓
check saved Storage versions
        ↓
Function.backward(...)
        ↓
parent gradients
        ↓
accumulate into Engine local_grads
```

After the **entire execution succeeds**:

```text
retain_graph=False
        ↓
release every reachable GradFn Context
```

This success-only release is intentional. If a hook or backward formula raises, the Engine does not leave the graph half released.

---

## 5. Shared graph nodes

Consider:

```python
x = Tensor(3.0, requires_grad=True)
shared = x * x
loss = (shared + shared).sum()
```

The graph is not a tree:

```text
        shared
       /      \
      /        \
     +----------+
          ↓
         sum
```

Topological traversal uses identity-based visited tracking, so `shared` is executed once.

Its incoming gradients are accumulated first:

```text
left contribution  +  right contribution
               ↓
        local_grads[shared]
               ↓
       execute shared backward once
```

This is why reverse-mode systems need graph-level accumulation rather than recursive "call parent immediately" logic.

---

## 6. Hooks

Register:

```python
handle = tensor.register_hook(fn)
```

Remove:

```python
handle.remove()
```

v0.8 hook pipeline:

```text
incoming grad
    ↓
hook 1
    ↓
hook 2
    ↓
...
    ↓
persistent accumulation + backward propagation
```

A hook receives a defensive copy of the gradient.

It may return:

```text
None
```

meaning "observe only", or a replacement gradient:

```python
def halve(grad):
    return grad * 0.5
```

Replacement gradients must have the same shape.

If a hook returns a Tensor, its device must match the hooked Tensor.

### Why transformed hooks affect parents

Hooks are executed before `Function.backward`.

So:

```text
hook transforms dL/dHidden
        ↓
Hidden.backward uses transformed dL/dHidden
        ↓
upstream leaf gradient changes
```

This makes hooks part of gradient flow rather than merely logging callbacks.

---

## 7. Leaf vs non-leaf gradient storage

Default engine behavior:

```text
leaf Tensor
  -> persistent .grad accumulation

non-leaf Tensor
  -> transient local gradient only
```

After:

```python
hidden.retain_grad()
```

behavior becomes:

```text
non-leaf Tensor
  -> transient local gradient
  -> plus persistent hidden.grad accumulation
```

Repeated retained graph executions accumulate both leaf and retained non-leaf gradients unless the user clears them.

---

## 8. Saved-version safety still applies

v0.4 introduced Storage version counters.

That invariant remains above the new Engine.

Forward:

```text
save tensor for backward
        ↓
remember Storage.version = N
```

If an alias changes the Storage:

```text
version N -> N+1
```

then backward performs:

```text
Context.check_saved_versions()
```

before the formula runs.

Graph lifetime and inplace safety are independent concepts:

```text
released Context
→ cannot reuse backward state

version mismatch
→ saved value is stale/unsafe
```

---

## 9. Custom `Function`

`Function` and `Context` are public teaching APIs in v0.8.

Example:

```python
class SquarePlusOne(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x + 1.0

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (grad_output * 2.0 * x,)
```

Usage:

```python
y = SquarePlusOne.apply(x)
```

No special Engine path exists for user-defined Functions.

They use exactly:

```text
Function.apply
   ↓
Context + GradFn
   ↓
AutogradEngine
```

That is the important architectural point.

---

## 10. CPU and EDU placement

The Engine does not ask:

```python
if tensor.device == "cpu": ...
elif tensor.device == "edu": ...
```

GradFn formulas currently operate on NumPy gradient arrays, while forward placement is already resolved by Tensor/Storage/Dispatcher.

Therefore the same engine semantics work for:

```text
CPU graph
EDU graph
CPU -> EDU transfer graph
```

Placement remains orthogonal to graph lifetime.

---

## 11. Why `create_graph=True` is not implemented

Current backward formulas look conceptually like:

```python
return grad_output * saved_numpy_array
```

The result is an `np.ndarray`.

That means backward computation itself is **not a Tensor graph**.

True higher-order gradients need:

```text
first backward
     ↓
Tensor-valued backward operations
     ↓
new GradFn graph
     ↓
second backward can differentiate it
```

v0.8 therefore does not fake support.

Calling:

```python
loss.backward(create_graph=True)
```

raises `NotImplementedError` with this architectural reason.

---

## 12. Recommended reading order

```text
mini_pytorch/autograd.py
        ↓
mini_pytorch/autograd_engine.py
        ↓
mini_pytorch/tensor.py
        ↓
tests/test_autograd_engine.py
        ↓
examples/11_autograd_engine.py
```

When reading, repeatedly ask:

1. What state belongs to Tensor?
2. What state belongs to GradFn/Context?
3. What state exists only during one Engine execution?
4. What gets released?
5. What persists and accumulates?

Those distinctions explain much of real-world Autograd behavior.