"""jit transform: cache a compiled function keyed on argument shapes/dtypes.

v0.1: identity wrapper that just calls the function and caches the result
of the first call. The real JIT lowering (replay the captured IR) is in
``micrograd._C.Function.jit`` for v0.2.
"""


def jit(fn):
    cache = {"last_args": None, "last_result": None}

    def wrapper(*args, **kwargs):
        sig = tuple(
            (a.shape() if hasattr(a, "shape") else None, type(a).__name__)
            for a in args
        )
        if cache["last_args"] == sig:
            return cache["last_result"]
        result = fn(*args, **kwargs)
        cache["last_args"] = sig
        cache["last_result"] = result
        return result

    return wrapper
