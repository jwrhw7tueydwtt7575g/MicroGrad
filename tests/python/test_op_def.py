"""Test the @op_def decorator and a custom loss."""
import micrograd as mg
from micrograd import tensor, nn, optim, losses
from micrograd.op_def import op_def


def test_op_def_decorator():
    print("test_op_def_decorator")
    @op_def(name="my_sq")
    def my_sq(x):
        return x * x

    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = my_sq(x)
    # The decorator wraps the function; in v0.1 the autograd is recorded
    # only if the function uses MicroGrad ops. x*x should record mul.
    # For v0.1 we just check the result is correct.
    assert out.tolist()[0] == 1.0
    print("  ok")


def test_custom_loss_runs():
    print("test_custom_loss_runs")
    @op_def(name="huber")
    def huber(pred, target):
        diff = pred - target
        abs_diff = diff.abs()
        out = []
        for v in abs_diff.tolist():
            if v < 1.0:
                out.append(0.5 * v * v)
            else:
                out.append(v - 0.5)
        return tensor(out, abs_diff.shape()).mean()

    from micrograd.data import sin_curve
    model = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
    opt = optim.Adam(model.parameters(), lr=1e-2)
    x, y = sin_curve(64)
    initial = None
    final = None
    for _ in range(20):
        opt.zero_grad()
        pred = model(x)
        loss = huber(pred, y)
        if initial is None:
            initial = loss.tolist()[0]
        loss.backward()
        opt.step()
        final = loss.tolist()[0]
    print(f"  huber loss: {initial:.4f} -> {final:.4f}")


if __name__ == "__main__":
    test_op_def_decorator()
    test_custom_loss_runs()
    print("op_def tests passed")
