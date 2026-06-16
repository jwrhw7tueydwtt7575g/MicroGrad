"""Tests for the Module system and optimizers."""
import micrograd as mg
from micrograd import nn, optim, losses, tensor


def test_linear_params():
    print("test_linear_params")
    layer = nn.Linear(3, 5)
    ps = layer.parameters()
    assert len(ps) == 2  # weight + bias
    assert ps[0].shape() == [3, 5]
    assert ps[1].shape() == [5]
    print("  ok")


def test_sequential_forward():
    print("test_sequential_forward")
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    x = tensor([[1.0, 2.0], [3.0, 4.0]])
    y = model(x)
    assert y.shape() == [2, 1]
    print("  ok")


def test_zero_grad():
    print("test_zero_grad")
    model = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 1))
    x = tensor([[1.0, 2.0]])
    y = tensor([[0.5]])
    opt = optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        opt.zero_grad()
        pred = model(x)
        loss = losses.mse(pred, y)
        loss.backward()
        # After backward, params with grad are set.
        grads = [p.grad().tolist() for p in model.parameters() if p.has_grad()]
        assert any(any(abs(g) > 0 for g in arr) for arr in grads)
        opt.step()
    print("  ok")


def test_adam_step_decreases_loss():
    print("test_adam_step_decreases_loss")
    from micrograd.data import sin_curve
    model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = optim.Adam(model.parameters(), lr=1e-2)
    x, y = sin_curve(64)
    losses_list = []
    for _ in range(50):
        opt.zero_grad()
        pred = model(x)
        loss = losses.mse(pred, y)
        loss.backward()
        opt.step()
        losses_list.append(loss.tolist()[0])
    assert losses_list[-1] < losses_list[0], (losses_list[0], losses_list[-1])
    print(f"  loss {losses_list[0]:.3f} -> {losses_list[-1]:.3f}")


if __name__ == "__main__":
    test_linear_params()
    test_sequential_forward()
    test_zero_grad()
    test_adam_step_decreases_loss()
    print("module/optim tests passed")
