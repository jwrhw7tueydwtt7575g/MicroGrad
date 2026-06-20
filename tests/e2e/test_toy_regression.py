"""E2E: train an MLP on a toy regression problem; loss must drop."""
import micrograd as mg
from micrograd import nn, optim, losses
from micrograd.data import sin_curve


def test_toy_regression():
    model = nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    opt = optim.Adam(model.parameters(), lr=3e-2)
    x, y = sin_curve(200)
    initial = None
    final = None
    for step in range(500):
        opt.zero_grad()
        pred = model(x)
        loss = losses.mse(pred, y)
        if initial is None:
            initial = loss.tolist()[0]
        loss.backward()
        opt.step()
        final = loss.tolist()[0]
    print(f"loss: {initial:.4f} -> {final:.4f}")
    assert final < 0.05, f"toy regression: final loss {final} too high"


if __name__ == "__main__":
    test_toy_regression()
    print("e2e toy regression passed")
