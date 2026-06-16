"""03_toy_regression: MLP fits sin(3x)."""
import micrograd as mg
from micrograd import nn, optim, losses
from micrograd.data import sin_curve

model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
opt = optim.Adam(model.parameters(), lr=1e-2)
x, y = sin_curve(200)

for step in range(500):
    opt.zero_grad()
    pred = model(x)
    loss = losses.mse(pred, y)
    loss.backward()
    opt.step()
    if step % 50 == 0 or step == 499:
        print(f"step {step:3d}: loss = {loss.tolist()[0]:.4f}")
