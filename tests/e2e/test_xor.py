"""E2E: train an MLP on XOR."""
import micrograd as mg
from micrograd import nn, optim, losses
from micrograd.data import xor


def test_xor():
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    opt = optim.Adam(model.parameters(), lr=5e-2)
    x, y = xor()
    initial = None
    final = None
    for step in range(500):
        opt.zero_grad()
        pred = model(x)
        loss = losses.cross_entropy(pred, y)
        if initial is None:
            initial = loss.tolist()[0]
        loss.backward()
        opt.step()
        final = loss.tolist()[0]
    print(f"xor loss: {initial:.4f} -> {final:.4f}")
    # Should learn XOR near zero loss.
    assert final < 0.05, f"xor: final loss {final} too high"


if __name__ == "__main__":
    test_xor()
    print("e2e xor passed")
