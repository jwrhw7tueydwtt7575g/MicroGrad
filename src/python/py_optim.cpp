#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "micrograd/optimizer.hpp"
#include "micrograd/sgd.hpp"
#include "micrograd/momentum.hpp"
#include "micrograd/adam.hpp"

namespace py = pybind11;
using namespace micrograd;

namespace {
std::vector<Tensor*> to_param_ptrs(std::vector<Tensor*> v) { return v; }
}

void register_optim(py::module& m) {
    py::class_<Optimizer>(m, "Optimizer")
        .def("step", &Optimizer::step)
        .def("zero_grad", &Optimizer::zero_grad)
        .def("lr", &Optimizer::lr)
        .def("set_lr", &Optimizer::set_lr);

    py::class_<SGD, Optimizer>(m, "SGD")
        .def(py::init([](std::vector<Tensor*> params, float lr, float momentum, float weight_decay) {
            return std::make_unique<SGD>(std::move(params), lr, momentum, weight_decay);
        }), py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.0f, py::arg("weight_decay") = 0.0f);

    py::class_<Momentum, Optimizer>(m, "Momentum")
        .def(py::init([](std::vector<Tensor*> params, float lr, float momentum, float weight_decay) {
            return std::make_unique<Momentum>(std::move(params), lr, momentum, weight_decay);
        }), py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.9f, py::arg("weight_decay") = 0.0f);

    py::class_<Adam, Optimizer>(m, "Adam")
        .def(py::init([](std::vector<Tensor*> params, float lr, float b1, float b2, float eps, float wd) {
            return std::make_unique<Adam>(std::move(params), lr, b1, b2, eps, wd);
        }), py::arg("params"), py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
            py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f);
}
