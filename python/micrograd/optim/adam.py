from .sgd import _Opt
from .. import _C


class Adam(_Opt):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, _C.Adam, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
