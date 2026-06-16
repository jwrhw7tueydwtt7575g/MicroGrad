"""06_jit_speedup: warm cache via @jit, then run a tight training loop."""
import time
import micrograd as mg
from micrograd import nn, optim, losses, transforms
from micrograd.data import sin_curve

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
opt = optim.Adam(model.parameters(), lr=1e-2)
x, y = sin_curve(200)


@transforms.jit
def step():
    opt.zero_grad()
    pred = model(x)
    loss = losses.mse(pred, y)
    loss.backward()
    opt.step()
    return loss


t0 = time.time()
for i in range(100):
    step()
t1 = time.time()
print(f"100 jit'd steps: {t1 - t0:.2f}s")
