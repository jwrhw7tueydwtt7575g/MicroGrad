"""Numerical gradient check for tensor ops."""
import math
import micrograd as mg
from micrograd import tensor


def _f(fn, *args, eps=1e-3):
    """Finite-difference gradient of fn at args. Each arg is a Tensor.

    For each i, perturb args[i] in-place, call fn with the perturbed tensor
    in its original position, and accumulate the per-element finite-difference
    gradient for that arg.

    IMPORTANT: all fn calls use **fresh, detached tensors** (requires_grad=False,
    no producer). This prevents the numeric perturbation calls from accidentally
    re-entering the autograd graph that was built during the analytic pass.
    """
    # Snapshot each arg as (flat_data, shape) — decoupled from any graph state.
    snapshots = [(a.tolist(), list(a.shape())) for a in args]

    out = []
    for i, (flat, shape) in enumerate(snapshots):
        flat = list(flat)  # mutable copy
        n = len(flat)
        grad = [0.0] * n
        for j in range(n):
            old = flat[j]

            # +eps perturbation: rebuild ALL tensors fresh (no requires_grad)
            flat[j] = old + eps
            slots_up = [tensor(list(snapshots[k][0]), snapshots[k][1])
                        if k != i else tensor(flat, shape)
                        for k in range(len(snapshots))]
            up = fn(*slots_up)

            # -eps perturbation
            flat[j] = old - eps
            slots_dn = [tensor(list(snapshots[k][0]), snapshots[k][1])
                        if k != i else tensor(flat, shape)
                        for k in range(len(snapshots))]
            dn = fn(*slots_dn)

            flat[j] = old
            up_v = up.tolist()[0] if hasattr(up, "tolist") else float(up)
            dn_v = dn.tolist()[0] if hasattr(dn, "tolist") else float(dn)
            grad[j] = (up_v - dn_v) / (2 * eps)
        out.append(grad)
    return out


def check(name, fn, *args_init, atol=1e-2):
    a0 = tensor(args_init[0], requires_grad=True)
    a1 = tensor(args_init[1], requires_grad=True) if len(args_init) > 1 else None
    out = fn(a0, a1) if a1 is not None else fn(a0)
    if not out.has_grad():
        out.init_grad()
    a0.zero_grad()
    if a1 is not None:
        a1.zero_grad()
    out.backward()
    analytic = a0.grad().tolist()
    if a1 is not None:
        analytic_b = a1.grad().tolist()
    # Pass raw init lists to _f so numeric uses fresh tensors internally
    args_for_numeric = (args_init[0], args_init[1]) if a1 is not None else (args_init[0],)
    numeric = _f(fn, *[tensor(a) for a in args_for_numeric])
    err_a = max(abs(analytic[k] - numeric[0][k]) for k in range(len(analytic)))
    msg = f"{name}: max |analytic - numeric| = {err_a:.4e}"
    if a1 is not None:
        err_b = max(abs(analytic_b[k] - numeric[1][k]) for k in range(len(analytic_b)))
        msg += f", b: {err_b:.4e}"
        assert err_b < atol, msg
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
    if not c.has_grad():
        c.init_grad()
    a.zero_grad()
    b.zero_grad()
    c.backward()
    ga = a.grad().tolist()
    gb = b.grad().tolist()
    # d/dA sum(A @ B) = ones_matrix @ B^T, where ones_matrix is same shape as A.
    # For A (2x2) and B (2x2): grad_A[i,j] = sum_k ones[i,k] * B[k,j]^T
    #   = sum_k B[j,k]  (since B^T[k,j] = B[j,k])
    # B rows: [0.5, -1.0], [2.0, 0.25]
    # grad_A row = [sum of B row 0, sum of B row 1] = [0.5+2.0, -1.0+0.25] = [2.5, -0.75]
    b_flat = b.tolist()
    b_rows = [[b_flat[0], b_flat[1]], [b_flat[2], b_flat[3]]]
    # grad_A[i,j] = sum of row j of B
    row0 = b_rows[0][0] + b_rows[0][1]  # 0.5 - 1.0 = -0.5
    row1 = b_rows[1][0] + b_rows[1][1]  # 2.0 + 0.25 = 2.25
    expected_a_flat = [row0, row1, row0, row1]   # same for both rows of A
    err_a = max(abs(ga[i] - expected_a_flat[i]) for i in range(4))
    assert err_a < 1e-3, f"matmul grad_A: err={err_a}, got {ga}, expected {expected_a_flat}"
    print(f"   max |da| = {err_a:.2e}")
    # d/dB sum(A @ B) = A^T @ ones_matrix
    # grad_B[i,j] = sum_k A[k,i]  (sum of column i of A)
    a_flat = a.tolist()
    a_rows = [[a_flat[0], a_flat[1]], [a_flat[2], a_flat[3]]]
    col0_a = a_rows[0][0] + a_rows[1][0]  # 1+3=4
    col1_a = a_rows[0][1] + a_rows[1][1]  # 2+4=6
    expected_b_flat = [col0_a, col0_a, col1_a, col1_a]   # [4,4,6,6]
    err_b = max(abs(gb[i] - expected_b_flat[i]) for i in range(4))
    assert err_b < 1e-3, f"matmul grad_B: err={err_b}, got {gb}, expected {expected_b_flat}"
    print(f"   max |db| = {err_b:.2e}")


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
