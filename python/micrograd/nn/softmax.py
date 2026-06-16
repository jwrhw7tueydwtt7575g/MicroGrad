from .module import Module


class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Simple softmax in pure-Python for v0.1.
        import math
        data = x.tolist()
        shape = x.shape()
        rank = len(shape)
        d = self.dim if self.dim >= 0 else self.dim + rank
        # Compute strides.
        strides = [1] * rank
        for i in range(rank - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        out = [0.0] * len(data)
        # outer, reduce, inner.
        inner = 1
        for i in range(d + 1, rank):
            inner *= shape[i]
        reduce = shape[d]
        outer = len(data) // (inner * reduce)
        for o in range(outer):
            for i in range(inner):
                mx = max(data[o * reduce * inner + r * inner + i] for r in range(reduce))
                s = 0.0
                exps = [0.0] * reduce
                for r in range(reduce):
                    e = math.exp(data[o * reduce * inner + r * inner + i] - mx)
                    exps[r] = e
                    s += e
                for r in range(reduce):
                    out[o * reduce * inner + r * inner + i] = exps[r] / s
        from .. import tensor as _t
        return _t(out, shape)
