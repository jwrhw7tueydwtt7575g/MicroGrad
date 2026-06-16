"""op_def decorator: register a Python forward+grad pair as a C++ op.

v0.1 path: the user's forward is run eagerly; the result is captured and
returned. The C++ side cannot yet lower arbitrary Python code to a kernel,
so this is essentially a Python-level op that lives outside the registry.
The @op_def decorator still gives a uniform interface.
"""
from ..tensor import tensor as _t


def op_def(name=None):
    def deco(fn):
        op_name = name or fn.__name__

        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            return out

        wrapper.__name__ = op_name
        wrapper.__qualname__ = op_name
        return wrapper

    return deco
