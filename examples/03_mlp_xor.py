"""A small nonlinear problem showing Module composition and Adam."""
import numpy as np

from mini_pytorch import Tensor, nn, optim

np.random.seed(42)

x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = Tensor([[0], [1], [1], [0]])

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

for step in range(2000):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 400 == 0:
        print(f"step={step:04d} loss={loss.item():.6f}")

print("predictions:\n", model(x).data)
