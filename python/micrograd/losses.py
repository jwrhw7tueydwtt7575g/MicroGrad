from .. import _C


def mse(pred, target):
    """Mean-squared-error loss."""
    return _C.mse_loss(pred, target)


def cross_entropy(logits, target):
    """Cross-entropy loss. `target` is one-hot, same shape as `logits`."""
    return _C.cross_entropy_loss(logits, target)


def bce(pred, target):
    """Binary cross-entropy loss."""
    return _C.bce_loss(pred, target)
