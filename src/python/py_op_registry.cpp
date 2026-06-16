#include <pybind11/pybind11.h>
#include "micrograd/op_registry.hpp"
#include "micrograd/op.hpp"

namespace py = pybind11;
using namespace micrograd;

void register_op_registry(py::module& m) {
    m.def("init_ops", &init_cpu_ops);
    m.def("has_op", [](const std::string& name) {
        return OpRegistry::instance().has(name, Device::cpu());
    });
}
