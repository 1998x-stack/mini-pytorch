# mini-pytorch v0.7 — Device / Backend Abstraction

> 这一版把“Tensor 住在哪里”正式变成框架的一等概念。

**版本主题：** Device → Backend-owned Storage → device-aware Dispatcher → Tensor.to

**Package version：** `0.7.0`

**当前验证基线：** `103/103 tests passed`

## 1. v0.7 相比 v0.6 增加了什么

v0.6 解决的是：

```text
复杂 NN operator
如何建立在 Dispatcher 上？
```

v0.7 继续解决：

```text
同一个 Tensor / operator API
如何支持不同 device/backend？
```

核心架构：

```text
                       Tensor
                         │
        shape / stride / dtype / device
                         │
                         ▼
                       Storage
                         │
                 backend-owned allocation
                         │
                         ▼
                    logical operator
                         │
                         ▼
                     Dispatcher
              ┌──────────┴──────────┐
              ▼                     ▼
           CPU key                EDU key
              │                     │
              └──── backend kernel ─┘
                         ▲
                         │
                   Autograd wrapper
```

这里的 `EDU` 是一个**模拟教育 backend**：底层仍使用 NumPy host memory，但拥有独立 Device、Backend 和 DispatchKey，因此可以在普通 CI 上验证真正的多 backend 路由逻辑。

它不是 CUDA，也不宣称 GPU 加速。

## 2. 新增 Device

```python
from mini_pytorch import Device

cpu = Device("cpu")
edu = Device("edu")
edu0 = Device("edu:0")
```

`Device` 是 immutable value object，支持：

- canonical string；
- equality；
- hash；
- index；
- validation。

当前仅支持：

```text
cpu
edu
```

## 3. Backend registry

新增：

```text
mini_pytorch/backend.py
```

抽象边界：

```text
Backend
├── allocate(size, dtype)
├── from_numpy(array)
└── to_numpy(allocation)
```

当前注册：

```text
cpu → NumPyBackend
edu → EducationalBackend
```

因此现在 Storage 不再直接假设：

```text
Storage == NumPy array
```

而是：

```text
Storage
├── Device
├── Backend
├── backend-owned allocation
└── VersionCounter
```

## 4. Tensor.device

```python
from mini_pytorch import Tensor

x = Tensor([1, 2, 3])
print(x.device)
# cpu

y = Tensor([1, 2, 3], device="edu")
print(y.device)
# edu
```

Tensor view 继续共享 Storage，因此也共享同一个 device。

## 5. Tensor.to(device)

### same device

```python
x.to("cpu") is x
# True
```

默认不复制。

如果：

```python
copy = x.to("cpu", copy=True)
```

则创建新的 Storage。

### cross-device

```python
y = x.to("edu")
```

会：

```text
source logical values
        ↓
new EDU Storage
        ↓
new Tensor
```

不会共享原 Storage。

如果源 Tensor 是 non-contiguous view，transfer 会按照**逻辑值**物化成新的 contiguous target Tensor。

## 6. Device copy 仍然支持 Autograd

```python
x = Tensor([2.0, -3.0], requires_grad=True)
y = x.to("edu")
loss = (y * y).sum()
loss.backward()
```

梯度仍会回到原始 CPU Tensor：

```text
x(cpu)
  ↓
CopyToDevice
  ↓
y(edu)
  ↓
y*y
  ↓
sum
  ↓
backward
  ↓
x.grad
```

v0.7 的 device copy backward 是 identity gradient：

```text
dL/dx = dL/dy
```

因为这一版不做 dtype conversion。

## 7. Dispatcher 现在真正 device-aware

CPU Tensor：

```text
DispatchKeySet {CPU}
```

EDU Tensor：

```text
DispatchKeySet {EDU}
```

requires-grad CPU：

```text
{Autograd, CPU}
```

requires-grad EDU：

```text
{Autograd, EDU}
```

Autograd wrapper 只 remove：

```text
Autograd
```

然后 redispatch，因此 backend identity 会保留：

```text
{Autograd, EDU}
      ↓
remove Autograd
      ↓
{EDU}
      ↓
EDU registration
```

## 8. Mixed-device operation 会明确失败

```python
cpu = Tensor([1.0], device="cpu")
edu = Tensor([2.0], device="edu")

cpu + edu
```

不会偷偷 copy，而是：

```text
DispatchError:
operator received tensors on different devices
```

这一步发生在 backend math 之前。

## 9. Scalar promotion 会继承 owning device

以前：

```python
x + 3.0
```

如果无脑把 `3.0` 包成默认 Tensor，很容易偷偷创建 CPU Tensor。

现在：

```python
x_edu + 3.0
```

会把 scalar 包装到：

```text
edu
```

同样：

```python
2.0 * x_edu
```

也保持 EDU route。

## 10. 旧的 Function.apply 路径也保持 device

v0.7 之前并不是所有 operator 都已经迁移到 Dispatcher。

例如：

```text
exp
log
div
部分 layout operator
```

仍走：

```text
Function.apply
```

v0.7 已经让这条路径也遵守：

```text
common input device
      ↓
output keeps device
```

mixed-device direct Function inputs 也会提前报错。

所以 device 语义不会依赖“这个 operator 是否已经迁入 Dispatcher”。

## 11. Module.to(device)

现在可以：

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, 3),
    nn.BatchNorm2d(8),
)

model.to("edu")
```

递归移动：

```text
Parameters
Buffers
child Modules
```

而 Parameter 对象本身不会被替换，因此 module registration 保持稳定。

建议和 PyTorch 常见工作流一样：

```text
先 model.to(device)
再创建 optimizer
```

## 12. CNN operator 在 EDU 上也可执行

当前这些 dispatched operators 都有 CPU + EDU registration：

```text
add
mul
matmul
relu
sum
conv2d
max_pool2d
```

EDU registration 当前复用同一组 NumPy reference math。

这并不意味着 CPU/EDU 是相同 backend，而是意味着：

> backend routing 已经独立于 numerical implementation。

未来真实 accelerator backend 可以替换其中一侧 kernel，而不需要修改 Tensor API。

## 13. 新增核心文件

```text
mini_pytorch/device.py
mini_pytorch/backend.py
tests/test_device_backend.py
docs/DEVICE_BACKEND.md
examples/10_device_backend.py
```

重点修改：

```text
mini_pytorch/storage.py
mini_pytorch/tensor.py
mini_pytorch/autograd.py
mini_pytorch/dispatcher.py
mini_pytorch/operator_registry.py
mini_pytorch/functional.py
mini_pytorch/nn/module.py
mini_pytorch/__init__.py
pyproject.toml
```

## 14. 推荐阅读顺序

```text
mini_pytorch/device.py
        ↓
mini_pytorch/backend.py
        ↓
mini_pytorch/storage.py
        ↓
mini_pytorch/tensor.py
        ↓
mini_pytorch/dispatcher.py
        ↓
mini_pytorch/operator_registry.py
        ↓
mini_pytorch/nn/module.py
        ↓
tests/test_device_backend.py
        ↓
docs/DEVICE_BACKEND.md
        ↓
examples/10_device_backend.py
```

## 15. 快速运行

```bash
pip install -e .
python -m unittest discover -s tests -v
```

运行 device/backend 示例：

```bash
python examples/10_device_backend.py
```

## 16. 当前验证基线

GitHub Actions：

```text
Python 3.11
NumPy 2.4.6

Ran 103 tests
OK
```

其中 v0.7 新增覆盖：

- Device parsing / equality / hash / validation；
- Backend registry；
- Storage.device；
- Tensor.device；
- same-device `.to()` no-op；
- same-device forced copy；
- cross-device Storage copy；
- alias break；
- differentiable transfer；
- CPU vs EDU DispatchKey；
- Autograd / NoGrad above EDU；
- mixed-device error；
- scalar promotion；
- direct Function.apply device preservation；
- Module.to Parameters/Buffers；
- Conv2d / MaxPool2d on EDU。

## 17. Explicit non-goals

v0.7 明确没有：

- real CUDA；
- CuPy；
- async copy；
- pinned memory；
- streams/events；
- mixed precision；
- caching allocator；
- distributed execution。

这些不是“漏做”，而是后续 runtime/device 深化版本的范围。

## 18. 下一版本

`v0.8` 应进入 **Advanced Autograd Engine**：

```text
graph lifetime
retain_grad
hooks
retain_graph
custom Function
higher-order gradient concepts
```

到这里，mini-pytorch 已经从：

```text
Tensor
```

逐步建立到：

```text
Tensor
+ Autograd
+ Optimizer
+ Data pipeline
+ Storage/views
+ Dispatcher
+ CNN operators
+ Device/Backend
```
