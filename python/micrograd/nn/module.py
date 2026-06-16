from .. import _C


class Module:
    """Base class for neural-network modules.

    Subclasses must implement ``forward``. The base class collects parameters
    from ``own_params`` and from registered children.
    """

    def __init__(self):
        self._children = []
        self._params = []  # list of (name, Tensor)

    def add_param(self, name, tensor):
        self._params.append((name, tensor))

    def parameters(self):
        out = [t for _, t in self._params]
        for c in self._children:
            out.extend(c.parameters())
        return out

    def named_parameters(self):
        out = list(self._params)
        for c in self._children:
            for n, t in c.named_parameters():
                out.append((n, t))
        return out

    def zero_grad(self):
        for p in self.parameters():
            if p.has_grad():
                p.zero_grad()

    def forward(self, x):  # pragma: no cover - subclasses override
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
