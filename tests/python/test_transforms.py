"""Tests for transforms: grad, vmap, jit."""
import micrograd as mg
from micrograd import tensor, transforms


def test_grad():
    print("test_grad")
    def f(x):
        return ((x * 2) ** 2).sum()

    g = transforms.grad(f)
    x = tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    dx = g(x)
    # d/dx (4x^2) = 8x
    expected = [8.0, 16.0, 24.0]
    got = dx[0].tolist()
    for e, v in zip(expected, got):
        assert abs(e - v) < 1e-3, (e, v)
    print("  ok")


def test_vmap():
    print("test_vmap")
    def f(x):
        return (x * 2).sum()

    f_batched = transforms.vmap(f, in_axes=0)
    x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = f_batched(x)
    assert out.shape() == [2], out.shape()
    vals = out.tolist()
    assert abs(vals[0] - 12.0) < 1e-3, vals
    assert abs(vals[1] - 30.0) < 1e-3, vals
    print("  ok")


def test_jit_caches():
    print("test_jit_caches")
    counter = [0]

    @transforms.jit
    def f(x):
        counter[0] += 1
        return x.sum()

    x = tensor([1.0, 2.0, 3.0])
    f(x); f(x); f(x)
    assert counter[0] == 1
    print("  ok")


if __name__ == "__main__":
    test_grad()
    test_vmap()
    test_jit_caches()
    print("transform tests passed")
