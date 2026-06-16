from .. import _C


class _Opt:
    """Common interface wrapping the C++ optimizer."""

    def __init__(self, params, lr, ctor, **kwargs):
        ptrs = list(params)
        self._opt = ctor(ptrs, lr, **kwargs)

    def step(self):
        self._opt.step()

    def zero_grad(self):
        self._opt.zero_grad()

    @property
    def lr(self):
        return self._opt.lr()

    @lr.setter
    def lr(self, v):
        self._opt.set_lr(v)


class SGD(_Opt):
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr, _C.SGD, momentum=momentum, weight_decay=weight_decay)
