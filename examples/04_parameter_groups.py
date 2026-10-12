"""Different learning rates for different parameter groups."""
from mini_pytorch import nn, optim

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

optimizer = optim.SGD(
    [
        {"params": model[0].parameters(), "lr": 1e-2},
        {"params": model[2].parameters(), "lr": 1e-3},
    ],
    lr=1e-2,
    momentum=0.9,
)

for i, group in enumerate(optimizer.param_groups):
    print(f"group {i}: lr={group['lr']}, parameters={len(group['params'])}")
