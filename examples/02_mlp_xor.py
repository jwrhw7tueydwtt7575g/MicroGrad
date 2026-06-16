"""02_mlp_xor: train a 2-layer MLP on XOR."""
import micrograd as mg
from micrograd import nn, optim, losses
from micrograd.data import xor

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
opt = optim.Adam(model.parameters(), lr=5e-2)
x, y = xor()

for step in range(500):
    opt.zero_grad()
    pred = model(x)
    loss = losses.cross_entropy(pred, y)
    loss.backward()
    opt.step()
    if step % 50 == 0 or step == 499:
        print(f"step {step:3d}: loss = {loss.tolist()[0]:.4f}")
