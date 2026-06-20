"""Minimal autograd smoke test.

Build two scalar tensors with `requires_grad=True`, compute `c = a * b + a`,
run backward, and print the resulting gradients on `a` and `b`.

For `c = a * b + a`:
    d c / d a = b + 1
    d c / d b = a
"""
import micrograd as mg
from micrograd import tensor


def main():
    a = tensor(2.0, requires_grad=True)
    b = tensor(5.0, requires_grad=True)

    c = a * b + a
    c.backward()

    print(f"a = {a.tolist()}, b = {b.tolist()}")
    print(f"c = a * b + a = {c.tolist()}")
    print(f"a.grad() = {a.grad().tolist()}  (expected {b.tolist()[0] + 1.0})")
    print(f"b.grad() = {b.grad().tolist()}  (expected {a.tolist()[0]})")


if __name__ == "__main__":
    main()