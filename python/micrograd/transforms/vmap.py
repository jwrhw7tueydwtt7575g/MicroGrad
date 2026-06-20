"""vmap transform: vectorize a function over a leading batch dim.

v0.1: naive implementation — loops the function over each batch slice.
This is the structure of the transform; a fused kernel path is v0.2.
"""


def vmap(fn, in_axes=0, out_axes=0):
    def wrapper(*args):
        # Find the batch size.
        axis = in_axes if isinstance(in_axes, int) else in_axes[0]
        if not isinstance(axis, int):
            raise NotImplementedError("vmap: v0.1 supports a single int in_axes")
        bs = args[0].shape()[axis]
        # Slice each arg along axis, run fn, stack outputs.
        outputs = []
        for i in range(bs):
            sliced = []
            for a in args:
                # Build a sliced view: data only (no autograd shim in v0.1).
                shape = list(a.shape())
                shape[axis] = 1
                flat = a.tolist()
                # Compute the slice manually: data is in row-major.
                rank = len(a.shape())
                # Compute strides.
                strides = [1] * rank
                for k in range(rank - 2, -1, -1):
                    strides[k] = strides[k + 1] * a.shape()[k + 1]
                # Linear index of the slice at batch i.
                base = 0
                coord = [0] * rank
                coord[axis] = i
                for k in range(rank):
                    base += coord[k] * strides[k]
                size = 1
                for k in range(rank):
                    if k != axis:
                        size *= a.shape()[k]
                # build a flat slice
                slc = [0.0] * size
                # Decode: linear index 0..size-1, mapping to (rank-dim) coords.
                for j in range(size):
                    tmp = j
                    full = 0
                    for k in range(rank):
                        if k == axis:
                            continue
                        # decode coord k
                        if k == rank - 1:
                            c = tmp
                        else:
                            s = 1
                            for kk in range(k + 1, rank):
                                if kk != axis:
                                    s *= a.shape()[kk]
                            c = tmp // s
                            tmp = tmp % s
                        full += c * strides[k]
                    slc[j] = flat[base + full]
                from .. import tensor as _t
                sliced.append(_t(slc, shape))
            out = fn(*sliced)
            outputs.append(out.tolist())
        # Stack along out_axes.
        if not outputs:
            return None
        flat_out = []
        for o in outputs:
            flat_out.extend(o)
        from .. import tensor as _t
        out_shape = list(out.shape())
        out_shape.insert(axis, bs)
        return _t(flat_out, out_shape)

    return wrapper
