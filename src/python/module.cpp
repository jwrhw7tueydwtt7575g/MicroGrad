// pybind11 module entry point for MicroGrad
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_tensor(py::module& m);
void register_function(py::module& m);
void register_module(py::module& m);
void register_optim(py::module& m);
void register_loss(py::module& m);
void register_op_registry(py::module& m);
void register_serializer(py::module& m);
void register_transforms(py::module& m);

PYBIND11_MODULE(_C, m) {
    m.doc() = "MicroGrad C++ core";
    register_op_registry(m);
    register_tensor(m);
    register_function(m);
    register_module(m);
    register_optim(m);
    register_loss(m);
    register_serializer(m);
    register_transforms(m);
}
