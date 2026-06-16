"""Toy regression/classification datasets for examples and tests."""
import math
from .. import tensor


def sin_curve(n=200):
    """y = sin(3x) on x in [-1, 1]. Returns (x, y) tensors."""
    import random
    random.seed(0)
    xs, ys = [], []
    for _ in range(n):
        x = random.uniform(-1.0, 1.0)
        xs.append([x])
        ys.append([math.sin(3.0 * x)])
    return tensor(xs), tensor(ys)


def xor():
    """XOR classification dataset. Returns (x, y_onehot)."""
    xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ys = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    return tensor(xs), tensor(ys)
