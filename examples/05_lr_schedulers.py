"""Compare a cosine learning-rate schedule without hiding the mechanics."""
from mini_pytorch import nn, optim

model = nn.Linear(2, 1)
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
scheduler = optim.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)

print("epoch  lr")
for epoch in range(11):
    print(f"{epoch:>5}  {optimizer.learning_rates[0]:.6f}")
    if epoch < 10:
        scheduler.step()
