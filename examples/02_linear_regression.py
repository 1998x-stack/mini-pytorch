"""Fit y = 3x - 2 using Linear + MSELoss + SGD."""
import numpy as np

from mini_pytorch import Tensor, nn, optim

np.random.seed(7)

x_np = np.linspace(-2, 2, 64).reshape(-1, 1)
y_np = 3.0 * x_np - 2.0
x = Tensor(x_np)
y = Tensor(y_np)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for step in range(120):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0 or step == 119:
        print(f"step={step:03d} loss={loss.item():.8f}")

print("weight:", model.weight.data.ravel())
print("bias:  ", model.bias.data)
