"""01_adder: simplest possible autograd example."""
import micrograd as mg
from micrograd import tensor

a = tensor([2.0], requires_grad=True)
b = tensor([-3.0], requires_grad=True)
c = tensor([10.0], requires_grad=True)

d = a * b + c
print("d =", d.tolist())
d.backward()
print("da =", a.grad().tolist(), "expected", (-3.0,))
print("db =", b.grad().tolist(), "expected", (2.0,))
print("dc =", c.grad().tolist(), "expected", (1.0,))
