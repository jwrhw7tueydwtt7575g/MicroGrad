#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "micrograd/function.hpp"
#include "micrograd/serializer.hpp"
#include "micrograd/active_fn.hpp"

namespace py = pybind11;
using namespace micrograd;

void register_function(py::module& m) {
    py::class_<Function, FunctionPtr>(m, "Function")
        .def(py::init<>())
        .def("save", &Function::save, py::arg("path"))
        .def_static("load", &Function::load, py::arg("path"))
        .def("backward", &Function::backward)
        .def("graph_output_id", [](const Function& f) { return f.graph().output_id; })
        .def("graph_size", [](const Function& f) { return static_cast<int>(f.graph().nodes.size()); });

    m.def("set_active_function", [](Function* f) { set_active_function(f); },
          py::arg("f"), "Set the active Function for autograd tracing. Pass None to clear.");
    m.def("active_function", &active_function, py::return_value_policy::reference);
}
