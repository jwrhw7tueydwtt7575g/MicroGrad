#include <pybind11/pybind11.h>
#include "micrograd/loss.hpp"

namespace py = pybind11;
using namespace micrograd;

void register_loss(py::module& m) {
    m.def("mse_loss", &mse_loss);
    m.def("cross_entropy_loss", &cross_entropy_loss);
    m.def("bce_loss", &bce_loss);
}
