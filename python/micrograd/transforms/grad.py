"""grad transform: differentiate a Python callable.

Usage:
    g = grad(lambda x: ((x * 2) ** 2).sum())
    dx = g(x_tensor)
"""
from .. import _C
from ..function import Function


def grad(fn, argnums=None):
    """Return a function that computes the gradient of `fn` w.r.t. its inputs.

    `argnums` selects which positional arguments to differentiate. Defaults
    to [0].
    """
    if argnums is None:
        argnums = [0]

    def wrapper(*args):
        for i in argnums:
            if i < len(args) and not args[i].requires_grad():
                args[i].requires_grad_(True)
            if i < len(args) and args[i].has_grad():
                args[i].zero_grad()
        with Function() as fn_obj:
            out = fn(*args)
            if isinstance(out, list):
                out = out[0]
            fn_obj.backward(out)
        return [args[i].grad() if i < len(args) else None for i in argnums]

    return wrapper
