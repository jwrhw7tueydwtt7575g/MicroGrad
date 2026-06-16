#include <pybind11/pybind11.h>
#include "micrograd/serializer.hpp"

namespace py = pybind11;
using namespace micrograd;

void register_serializer(py::module& m) {
    m.def("save_function", &Serializer::save_function);
    m.def("load_function", &Serializer::load_function, py::arg("path"));
    m.def("save_blob", &Serializer::save_blob);
    m.def("load_blob", &Serializer::load_blob);
    m.def("save_optimizer", &Serializer::save_optimizer);
    m.def("load_optimizer", &Serializer::load_optimizer);
}
