"""Generic training loop."""


def train(model, optimizer, loss_fn, x, y, steps=100, log_every=10,
          x_test=None, y_test=None, accuracy_fn=None):
    """Run a simple training loop.

    model: nn.Module
    optimizer: optim.*
    loss_fn: callable(pred, target) -> scalar Tensor
    x, y: input and target tensors
    steps: number of optimizer steps
    """
    for step in range(steps):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == steps - 1:
            msg = f"step {step:4d}: loss = {loss.tolist()[0]:.4f}"
            if x_test is not None and y_test is not None and accuracy_fn is not None:
                msg += f", acc = {accuracy_fn(model(x_test), y_test):.3f}"
            print(msg)
    return loss
