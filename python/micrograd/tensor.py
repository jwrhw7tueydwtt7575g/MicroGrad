"""Tensor construction helpers and context managers."""
from . import _C
from typing import Iterable, List


def tensor(data, shape=None, dtype="float32", requires_grad=False):
    """Construct a tensor from a Python list, numpy array, or (data, shape) pair.

    Examples
    --------
    >>> tensor([1.0, 2.0, 3.0])
    >>> tensor([[1, 2], [3, 4]])
    >>> tensor([1, 2, 3, 4], shape=[2, 2])
    """
    import numpy as np
    if isinstance(data, _C.Tensor):
        if requires_grad:
            data.requires_grad_(True)
        return data
    arr = np.asarray(data, dtype=np.float32)
    flat = arr.reshape(-1).tolist()
    s = list(arr.shape) if shape is None else list(shape)
    t = _C.Tensor(flat, s)
    if requires_grad:
        t.requires_grad_(True)
    return t


def _is_grad_enabled() -> bool:
    # The C++ side exposes a thread-local Tracer; we model enable/disable via
    # a simple Python flag plus the no_grad() ctx below.
    return _grad_enabled


_grad_enabled = True


class _Ctx:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._prev = None

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = self.enabled
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev


def no_grad():
    """Context manager that disables gradient recording."""
    return _Ctx(False)


def enable_grad():
    """Context manager that re-enables gradient recording."""
    return _Ctx(True)
