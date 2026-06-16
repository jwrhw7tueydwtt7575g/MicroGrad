"""Python-side Function wrapper. Manages the active graph during a step.

Each call to ``Function()`` creates a fresh C++ Function and activates it as
the tracer's current graph. Ops recorded during the call populate the
graph. ``backward()`` runs the captured graph in reverse.

This is a context manager so a user can also enter it explicitly:

    with mg.Function() as fn:
        out = model(x)
        loss = losses.mse(out, y)
        fn.backward(loss)
"""
from contextlib import contextmanager
from . import _C


class Function:
    """Wraps a C++ Function. Each instance is one training step."""

    def __init__(self):
        self._impl = _C.Function()
        self._active = False
        self._prev = None

    def __enter__(self):
        # Push ourselves as the active function on the thread-local tracer.
        # The C++ side does this via Function::Scope; we approximate by
        # storing a global here and the C++ side reads it through
        # _C.set_active_function(self._impl).
        if not hasattr(_C, "set_active_function"):
            raise RuntimeError("C++ _C.set_active_function not available; rebuild extension")
        _C.set_active_function(self._impl)
        self._active = True
        return self

    def __exit__(self, *args):
        _C.set_active_function(None)
        self._active = False

    def backward(self, loss):
        """Run backward starting from `loss`."""
        if not self._active:
            # Auto-activate (eager backward).
            with self:
                self._impl.backward(loss)
        else:
            self._impl.backward(loss)

    def save(self, path):
        self._impl.save(path)

    @staticmethod
    def load(path):
        f = Function()
        f._impl = _C.Function.load(path)
        return f

    @property
    def graph_size(self):
        return self._impl.graph_size()
