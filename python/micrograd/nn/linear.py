from .. import _C
from .module import Module


class Linear(Module):
    """Fully-connected linear layer y = x @ W + b.

    Parameters
    ----------
    in_features : int
    out_features : int
    bias : bool (default True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        import numpy as np
        # Kaiming-uniform init.
        bound = (1.0 / in_features) ** 0.5
        rng = np.random.default_rng(42)
        w_data = rng.uniform(-bound, bound, size=(in_features, out_features)).astype(np.float32)
        w = _C.Tensor(w_data.reshape(-1).tolist(), [in_features, out_features])
        w.requires_grad_(True)
        self.add_param("weight", w)
        if bias:
            b = _C.Tensor([0.0] * out_features, [out_features])
            b.requires_grad_(True)
            self.add_param("bias", b)

    def forward(self, x):
        y = x @ self._params[0][1]
        if len(self._params) > 1:
            y = y + self._params[1][1]
        return y
