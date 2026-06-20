#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "micrograd/serializer.hpp"

namespace py = pybind11;
using namespace micrograd;

void register_serializer(py::module& m) {
    m.def("save_function", &Serializer::save_function);
    m.def("load_function", &Serializer::load_function, py::arg("path"));
    m.def("save_blob", &Serializer::save_blob);
    m.def("load_blob", [](const std::string& dir, const std::string& hash,
                          const std::vector<int64_t>& shape, const std::string& dtype) {
        DType dt = DType::F32;
        if (dtype != "float32") {
            throw std::runtime_error("load_blob: unsupported dtype: " + dtype);
        }
        return Serializer::load_blob(dir, hash, Shape(shape), dt, Device::cpu());
    });
    m.def("save_optimizer", &Serializer::save_optimizer);
    m.def("load_optimizer", &Serializer::load_optimizer);
}
