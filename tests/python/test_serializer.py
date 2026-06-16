"""Test the save/load round-trip for Function and Optimizer."""
import os
import tempfile
import micrograd as mg
from micrograd import _C, nn, optim, losses, tensor


def test_function_save_load():
    print("test_function_save_load")
    f = _C.Function()
    # The v0.1 path: graph is empty; just confirm save/load round-trips.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.json")
        _C.save_function(f, path)
        f2 = _C.Function.load(path)
        assert f2.graph_size() == f.graph_size()
    print("  ok")


def test_optimizer_save_load():
    print("test_optimizer_save_load")
    layer = nn.Linear(2, 2)
    opt = optim.SGD(layer.parameters(), lr=0.05)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "opt.json")
        _C.save_optimizer(opt._opt, path)
        opt.lr = 0.99  # change
        _C.load_optimizer(opt._opt, path)
        assert abs(opt.lr - 0.05) < 1e-6, opt.lr
    print("  ok")


def test_blob_save_load():
    print("test_blob_save_load")
    import numpy as np
    data = [1.0, 2.0, 3.0, 4.0]
    t = tensor(data, [2, 2])
    with tempfile.TemporaryDirectory() as d:
        h = _C.save_blob(t, d)
        t2 = _C.load_blob(d, h, [2, 2], "float32")
        assert t2.tolist() == data
    print("  ok")


if __name__ == "__main__":
    test_function_save_load()
    test_optimizer_save_load()
    test_blob_save_load()
    print("serializer tests passed")
