#include <pybind11/pybind11.h>
#include "micrograd/function.hpp"

namespace py = pybind11;
using namespace micrograd;

void register_transforms(py::module& m) {
    m.def("grad_transform", [](Function& f, std::vector<int> argnums) {
        return f.grad(std::move(argnums));
    });
    m.def("vmap", [](Function& f, int in_b, int out_b) { return f.vmap(in_b, out_b); });
    m.def("jit", [](Function& f) { return f.jit(); });
}
