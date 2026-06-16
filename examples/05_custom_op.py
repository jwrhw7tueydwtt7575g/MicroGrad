"""05_custom_op: register a custom op via @op_def."""
import micrograd as mg
from micrograd import nn, optim, losses, tensor
from micrograd.op_def import op_def


@op_def(name="smooth_l1")
def smooth_l1(pred, target):
    """Smooth L1 (Huber) loss — less sensitive to outliers than MSE."""
    diff = pred - target
    abs_diff = diff.abs()
    # In Python for v0.1.
    import math
    out = []
    for x in abs_diff.tolist():
        if x < 1.0:
            out.append(0.5 * x * x)
        else:
            out.append(x - 0.5)
    return tensor(out, abs_diff.shape).mean()


# Use it as a loss for a tiny regression problem.
from micrograd.data import sin_curve

model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
opt = optim.Adam(model.parameters(), lr=1e-2)
x, y = sin_curve(200)

for step in range(300):
    opt.zero_grad()
    pred = model(x)
    loss = smooth_l1(pred, y)
    loss.backward()
    opt.step()
    if step % 50 == 0 or step == 299:
        print(f"step {step:3d}: smooth_l1 = {loss.tolist()[0]:.4f}")
