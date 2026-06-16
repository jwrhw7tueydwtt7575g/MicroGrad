from .sgd import SGD, _Opt
from .. import _C


class Momentum(_Opt):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0):
        super().__init__(params, lr, _C.Momentum, momentum=momentum, weight_decay=weight_decay)
