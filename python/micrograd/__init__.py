"""MicroGrad — a small autograd library with a C++ core and pybind11 bindings."""
from . import _C
from . import nn
from . import optim
from . import transforms
from . import losses
from . import op_def
from .function import Function
from .tensor import tensor, no_grad, enable_grad
from .train.loop import train

__all__ = ["tensor", "no_grad", "enable_grad", "nn", "optim", "transforms",
           "losses", "op_def", "Function", "train"]
__version__ = "0.1.1"

# Register all built-in CPU ops at import time.
_C.init_ops()
