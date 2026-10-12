# mini-pytorch v0.8 — Advanced Autograd Engine

> 把 `backward()` 从 Tensor 内部的一段 DFS，升级成一个有显式生命周期、hook 与 retained-gradient 语义的 Autograd Engine。

**Package version:** `0.8.0`

**Validated release-candidate baseline:** `123/123 tests passed`

## 1. 本版本解决什么问题

v0.7 已经把 Tensor placement 拆成：

```text
Tensor / Storage / Device / Backend / Dispatcher
```

v0.8 继续回答：

> 动态计算图在 backward 时到底是谁执行、哪些梯度只是临时值、哪些梯度要永久保存、图什么时候释放、为什么第二次 backward 会报错？

核心结构：

```text
forward Tensor graph
        ↓
  GradFn / Context
        ↓
   AutogradEngine
 ┌──────┼───────────────┐
 ▼      ▼               ▼
VJP   graph lifetime   hooks
 │    retain_graph      │
 ▼                      ▼
leaf grad        retained non-leaf grad
```

## 2. 相比 v0.7 的核心变化

v0.7：

```text
Tensor.backward()
    ↓
Tensor 内部 DFS
    ↓
直接执行 Function.backward
```

v0.8：

```text
Tensor.backward()
    ↓
AutogradEngine.backward()
    ↓
构建 topology
    ↓
检查 graph availability
    ↓
local gradient accumulation
    ↓
hooks
    ↓
retain_grad / leaf grad
    ↓
Function.backward VJP
    ↓
release or retain graph
```

因此 `Tensor.backward()` 现在只是用户入口，执行策略属于 Engine。

## 3. Graph lifetime

默认：

```python
loss.backward()
```

成功后会释放所有可达 `Context` 的 backward-only 状态：

```text
saved_tensors
saved versions
forward input references
```

Tensor 本身的数据和 graph structural references 仍然存在，但 Context 被标记为 released。

所以：

```python
loss.backward()
loss.backward()
```

第二次会明确报错。

这解释了真实 PyTorch 中“Trying to backward through the graph a second time”背后的生命周期概念。

## 4. `retain_graph=True`

```python
loss.backward(retain_graph=True)
loss.backward()
```

第一次保留 Context；第二次继续使用并在最后释放。

注意：

```text
retain_graph
!=
zero_grad
```

`retain_graph` 控制**图状态能否再次执行**；`.grad` 是否累加是另一件事。

例如两次 backward 后 leaf gradient 会继续累加，除非你显式清零。

## 5. Engine-local gradient vs `Tensor.grad`

v0.8 明确区分：

```text
local_grads
```

和：

```text
Tensor.grad
```

### Engine local gradient

只在一次 backward 内存在：

```text
local_grads[id(tensor)] -> ndarray
```

用于 shared-node 梯度合并和 backward 传播。

### Persistent Tensor.grad

默认只有 leaf Tensor 会持久化 `.grad`。

这两个状态不应该混成一个概念。

## 6. Non-leaf `retain_grad()`

默认：

```python
x = Tensor([2.0, 3.0], requires_grad=True)
hidden = x * x
hidden.sum().backward()

print(hidden.grad)
# None
```

因为 `hidden` 是 non-leaf。

现在可以：

```python
hidden.retain_grad()
```

之后：

```text
Engine local gradient
      ↓
copy/accumulate
      ↓
hidden.grad
```

这非常适合调试中间层梯度。

## 7. Gradient hooks

现在支持：

```python
handle = tensor.register_hook(fn)
handle.remove()
```

hook pipeline：

```text
incoming gradient
      ↓
hook 1
      ↓
hook 2
      ↓
...
      ↓
Tensor.grad accumulation
      ↓
Function.backward
```

hook 可以只观察：

```python
def observe(grad):
    print(grad)
```

也可以返回新的 gradient：

```python
def halve(grad):
    return grad * 0.5
```

返回值会继续参与上游梯度传播。

v0.8 约束：

- hooks 按注册顺序执行；
- hook 输入是 defensive copy；
- 返回 `None` 表示不修改；
- replacement shape 必须一致；
- 如果返回 Tensor，它的 device 必须一致；
- handle.remove() 可重复调用。

## 8. Shared-node accumulation

例如：

```python
shared = x * x
loss = (shared + shared).sum()
```

不是一棵树，而是 DAG：

```text
        shared
       /      \
      /        \
     +----------+
          ↓
         sum
```

Engine 会先合并来自两个 consumer 的梯度：

```text
left grad + right grad
        ↓
local_grads[shared]
        ↓
shared backward 只执行一次
```

这比递归“拿到梯度就立即往父节点传”更接近真正 Autograd Engine 的思维方式。

## 9. Custom `Function`

v0.8 正式把：

```python
Function
Context
```

作为公开教学接口。

示例：

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

调用：

```python
y = SquarePlusOne.apply(x)
```

它不会走特殊路径，而是完全复用：

```text
Function.apply
    ↓
Context + GradFn
    ↓
AutogradEngine
```

并自动获得：

- saved-version safety；
- graph lifetime；
- retain_graph；
- CPU/EDU placement preservation。

## 10. Saved-version safety 仍然有效

v0.4 的 inplace version counter 没有被 v0.8 重构破坏。

Forward：

```text
save Tensor for backward
      ↓
remember Storage.version
```

alias inplace mutation：

```text
version N -> N+1
```

Backward 在执行公式前：

```text
Context.check_saved_versions()
```

发现不一致立即报错。

因此：

```text
graph lifetime
```

和：

```text
inplace safety
```

是两个正交机制。

## 11. CPU / EDU 行为保持一致

Autograd Engine 不包含：

```python
if device == "cpu": ...
if device == "edu": ...
```

因此同样的 graph lifetime、hook、retain_grad、retain_graph 语义可以运行在：

```text
CPU graph
EDU graph
CPU -> EDU transfer graph
```

Device/Backend 仍由 v0.7 的 Storage + Dispatcher 体系负责。

## 12. 为什么不实现假的 `create_graph=True`

当前 backward 公式返回：

```text
np.ndarray
```

而不是：

```text
Tensor
```

因此 backward 本身不会形成新的计算图。

真正二阶导数需要：

```text
backward formulas
    ↓
Tensor operators
    ↓
new GradFn graph
    ↓
second backward
```

所以本版本调用：

```python
loss.backward(create_graph=True)
```

会明确：

```text
NotImplementedError
```

而不是假装支持 higher-order gradients。

## 13. 新增核心文件

```text
mini_pytorch/autograd_engine.py
tests/test_autograd_engine.py
docs/AUTOGRAD_ENGINE.md
examples/11_autograd_engine.py
```

重点修改：

```text
mini_pytorch/autograd.py
mini_pytorch/tensor.py
mini_pytorch/__init__.py
pyproject.toml
```

## 14. 推荐阅读顺序

```text
mini_pytorch/autograd.py
        ↓
mini_pytorch/autograd_engine.py
        ↓
mini_pytorch/tensor.py
        ↓
tests/test_autograd_engine.py
        ↓
docs/AUTOGRAD_ENGINE.md
        ↓
examples/11_autograd_engine.py
```

## 15. 快速运行

安装：

```bash
pip install -e .
```

完整测试：

```bash
python -m unittest discover -s tests -v
```

当前 release-candidate 基线：

```text
123 tests
OK
```

运行教学示例：

```bash
python examples/11_autograd_engine.py
```

## 16. v0.8 新增测试覆盖

新增 20 个 Autograd Engine tests，覆盖：

- default graph release；
- repeated backward error；
- retain_graph reuse；
- shared-node release；
- CPU/EDU lifetime parity；
- non-leaf default grad behavior；
- retain_grad；
- retained-grad accumulation；
- hook observe；
- hook transform；
- hook ordering；
- removable handle；
- invalid hook shape；
- hook Tensor device mismatch；
- custom Function；
- custom Function version safety；
- custom Function EDU placement；
- Tensor gradient seed device check；
- create_graph explicit boundary。

旧 103 个测试同时全部通过。

## 17. Explicit non-goals

v0.8 不实现：

- true second derivatives；
- full `create_graph=True`；
- forward-mode AD；
- parallel Autograd workers；
- anomaly detection；
- stream-aware CUDA engine；
- DDP gradient buckets；
- compiler graph capture。

## 18. 下一版本

下一版 `v0.9` 将进入 **Training Runtime / Serialization / Reproducibility**：

```text
model state
optimizer state
scheduler state
RNG state
epoch / global step
      ↓
unified checkpoint
      ↓
deterministic resume
```

它会建立在 v0.8 已经明确的 graph/runtime 边界之上。