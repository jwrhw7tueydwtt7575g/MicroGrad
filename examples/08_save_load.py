"""08_save_load: train, save, load, verify."""
import os
import tempfile
import micrograd as mg
from micrograd import _C, nn, optim, losses, tensor
from micrograd.data import sin_curve

model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)
opt = optim.Adam(model.parameters(), lr=1e-2)
x, y = sin_curve(100)

for step in range(50):
    opt.zero_grad()
    pred = model(x)
    loss = losses.mse(pred, y)
    loss.backward()
    opt.step()

print(f"trained loss: {loss.tolist()[0]:.4f}")

with tempfile.TemporaryDirectory() as d:
    # Save a function (the IR; v0.1 doesn't save tensor data).
    f = _C.Function()
    _C.save_function(f, os.path.join(d, "graph.json"))
    f2 = _C.Function.load(os.path.join(d, "graph.json"))
    print(f"saved/loaded function graph size = {f2.graph_size()}")
    # Save an optimizer.
    _C.save_optimizer(opt._opt, os.path.join(d, "opt.json"))
    _C.load_optimizer(opt._opt, os.path.join(d, "opt.json"))
    print(f"saved/loaded optimizer lr = {opt.lr}")
