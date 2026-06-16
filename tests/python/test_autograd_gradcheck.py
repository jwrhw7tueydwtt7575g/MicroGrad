"""Numerical gradient check for tensor ops."""
import math
import micrograd as mg
from micrograd import tensor


def _f(fn, *args, eps=1e-3):
    """Finite-difference gradient of fn at args."""
    out = []
    for i, a in enumerate(args):
        flat = a.tolist()
        shape = a.shape()
        n = len(flat)
        grad = [0.0] * n
        for j in range(n):
            old = flat[j]
            flat[j] = old + eps
            up = fn(tensor(flat, shape), *[args[k] for k in range(i + 1, len(args))])
            flat[j] = old - eps
            dn = fn(tensor(flat, shape), *[args[k] for k in range(i + 1, len(args))])
            flat[j] = old
            up_v = up.tolist()[0] if hasattr(up, "tolist") else up
            dn_v = dn.tolist()[0] if hasattr(dn, "tolist") else dn
            grad[j] = (up_v - dn_v) / (2 * eps)
        out.append(grad)
    return out


def check(name, fn, *args_init, atol=1e-2):
    a0 = tensor(args_init[0], requires_grad=True)
    a1 = tensor(args_init[1], requires_grad=True) if len(args_init) > 1 else None
    out = fn(a0, a1) if a1 is not None else fn(a0)
    out.init_grad() if not out.has_grad() else None
    a0.zero_grad()
    if a1 is not None:
        a1.zero_grad()
    out.backward()
    analytic = a0.grad().tolist()
    if a1 is not None:
        analytic_b = a1.grad().tolist()
    numeric = _f(fn, *(args_init))
    err_a = max(abs(analytic[k] - numeric[0][k]) for k in range(len(analytic)))
    msg = f"{name}: max |analytic - numeric| = {err_a:.4e}"
    if a1 is not None:
        err_b = max(abs(analytic_b[k] - numeric[1][k]) for k in range(len(analytic_b)))
        msg += f", b: {err_b:.4e}"
    assert err_a < atol, msg
    print("  ", msg)


def test_add():
    print("add:"); check("add", lambda a, b: (a + b).sum(), [1.0, 2.0, 3.0], [0.5, -1.0, 4.0])


def test_sub():
    print("sub:"); check("sub", lambda a, b: (a - b).sum(), [1.0, 2.0, 3.0], [0.5, -1.0, 4.0])


def test_mul():
    print("mul:"); check("mul", lambda a, b: (a * b).sum(), [1.0, 2.0, 3.0], [0.5, -1.0, 4.0])


def test_div():
    print("div:"); check("div", lambda a, b: (a / b).sum(), [1.0, 2.0, 3.0], [0.5, 1.0, 4.0])


def test_pow():
    print("pow:"); check("pow", lambda a, b: (a ** b).sum(), [1.0, 2.0, 0.5], [0.5, 1.0, 2.0])


def test_relu():
    print("relu:"); check("relu", lambda a: a.relu().sum(), [1.0, -2.0, 3.0])


def test_sigmoid():
    print("sigmoid:"); check("sigmoid", lambda a: a.sigmoid().sum(), [0.5, -1.0, 2.0])


def test_tanh():
    print("tanh:"); check("tanh", lambda a: a.tanh().sum(), [0.5, -1.0, 2.0])


def test_matmul():
    print("matmul:")
    a = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = tensor([[0.5, -1.0], [2.0, 0.25]], requires_grad=True)
    c = (a @ b).sum()
    c.init_grad() if not c.has_grad() else None
    a.zero_grad()
    b.zero_grad()
    c.backward()
    ga = a.grad().tolist()
    gb = b.grad().tolist()
    # analytic: d/dA sum(A@B) = ones @ B^T = [[1,1]@[[0.5,2],[-1,0.25]]] = [-0.5, 2.25],[...]
    expected_a = [sum(b.tolist()[k]) for k in range(2)]   # ones(2) @ B
    expected_a_full = [expected_a, expected_a]
    err = max(abs(ga[i] - expected_a_full[i // 2][i % 2]) for i in range(4))
    assert err < 1e-3, f"matmul: {err}"
    print("   max |da| =", err)


if __name__ == "__main__":
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_pow()
    test_relu()
    test_sigmoid()
    test_tanh()
    test_matmul()
    print("all gradchecks passed")
