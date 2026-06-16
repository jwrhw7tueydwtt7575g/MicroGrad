#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "micrograd/module.hpp"
#include "micrograd/linear.hpp"
#include "micrograd/sequential.hpp"
#include "micrograd/relu.hpp"
#include "micrograd/softmax.hpp"
#include "micrograd/tensor.hpp"

namespace py = pybind11;
using namespace micrograd;

namespace {
Tensor call_module(Module& m, const Tensor& x) { return m.forward(x); }
}

void register_module(py::module& m) {
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &call_module)
        .def("parameters", &Module::parameters, py::return_value_policy::reference_internal)
        .def("zero_grad", &Module::zero_grad)
        .def("state_dict", &Module::state_dict)
        .def("load_state_dict", &Module::load_state_dict)
        .def("__call__", &call_module);

    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int, bool>(), py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("weight", [](Linear& l) -> Tensor& { return l.weight(); }, py::return_value_policy::reference_internal)
        .def("bias",   [](Linear& l) -> Tensor& { return l.bias(); },   py::return_value_policy::reference_internal);

    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(py::init<>())
        .def("add", &Sequential::add)
        .def("__call__", &call_module);

    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>())
        .def("__call__", &call_module);

    py::class_<Softmax, Module, std::shared_ptr<Softmax>>(m, "Softmax")
        .def(py::init<int>(), py::arg("dim") = -1)
        .def("__call__", &call_module);
}
