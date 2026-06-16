// pybind11 tensor binding
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "micrograd/tensor.hpp"
#include "micrograd/op_registry.hpp"
#include "micrograd/ir.hpp"
#include "micrograd/autograd.hpp"
#include "micrograd/function.hpp"

namespace py = pybind11;
using namespace micrograd;

namespace {

Tensor py_add(const Tensor& a, const Tensor& b) { return run_op("add", {a, b})[0]; }
Tensor py_sub(const Tensor& a, const Tensor& b) { return run_op("sub", {a, b})[0]; }
Tensor py_mul(const Tensor& a, const Tensor& b) { return run_op("mul", {a, b})[0]; }
Tensor py_div(const Tensor& a, const Tensor& b) { return run_op("div", {a, b})[0]; }
Tensor py_pow(const Tensor& a, const Tensor& b) { return run_op("pow", {a, b})[0]; }
Tensor py_matmul(const Tensor& a, const Tensor& b) { return run_op("matmul", {a, b})[0]; }
Tensor py_neg(const Tensor& a) { return run_op("neg", {a})[0]; }
Tensor py_relu(const Tensor& a) { return run_op("relu", {a})[0]; }
Tensor py_exp(const Tensor& a) { return run_op("exp", {a})[0]; }
Tensor py_log(const Tensor& a) { return run_op("log", {a})[0]; }
Tensor py_abs(const Tensor& a) { return run_op("abs", {a})[0]; }
Tensor py_sigmoid(const Tensor& a) { return run_op("sigmoid", {a})[0]; }
Tensor py_tanh(const Tensor& a) { return run_op("tanh", {a})[0]; }
Tensor py_sum(const Tensor& a) { return run_op("sum", {a})[0]; }
Tensor py_mean(const Tensor& a) { return run_op("mean", {a})[0]; }

void py_backward(Tensor& t) {
    // If the tensor has a producer, use its graph.
    IRNode* prod = t.producer();
    if (!prod) {
        // Nothing to backprop into; tensor is a leaf with no grad hook.
        return;
    }
    Function f(prod->id >= 0 ? Graph{} : Graph{});
    // Walk back to find the Function that owns the graph. The thread-local
    // Tracer's function is the source of truth.
    Tracer& tr = Tracer::current();
    if (!tr.function) {
        // Build an ad-hoc Function wrapping the producer's graph.
        // Easiest: the producer's graph is whatever the tracer recorded
        // into. We rely on the Python side to manage this: the
        // py_function.Function is the active Function during the call.
        // The cleanest path: just call f.backward(t) with the function
        // from the tracer. The Python wrapper is responsible for setting
        // up the active Function — for raw C++ tests, we create one.
        Function local;
        // Copy the tracer's graph by re-anchoring through producer pointer.
        // In v0.1 we rely on the Python path for backward; provide a
        // working stub that no-ops if no Function is active.
        return;
    }
    tr.function->backward(t);
}

}  // namespace

void register_tensor(py::module& m) {
    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init([](const std::vector<int64_t>& shape, const std::string& dtype,
                         bool requires_grad) {
            (void)dtype;
            Tensor t = Tensor::empty(Shape(shape), DType::F32, Device::cpu());
            if (requires_grad) t.set_requires_grad(true);
            return t;
        }), py::arg("shape"), py::arg("dtype") = "float32", py::arg("requires_grad") = false)
        .def(py::init([](std::vector<float> data, const std::vector<int64_t>& shape) {
            Tensor t = Tensor::empty(Shape(shape), DType::F32, Device::cpu());
            int64_t n = t.numel();
            if (static_cast<int64_t>(data.size()) != n) {
                throw std::runtime_error("data size does not match shape numel");
            }
            std::memcpy(t.data_ptr(), data.data(), static_cast<size_t>(n) * sizeof(float));
            return t;
        }), py::arg("data"), py::arg("shape"))
        .def("__add__", &py_add)
        .def("__radd__", &py_add)
        .def("__sub__", &py_sub)
        .def("__rsub__", [](const Tensor& a, const Tensor& b) { return py_sub(b, a); })
        .def("__mul__", &py_mul)
        .def("__rmul__", &py_mul)
        .def("__truediv__", &py_div)
        .def("__neg__", &py_neg)
        .def("__pow__", &py_pow)
        .def("__matmul__", &py_matmul)
        .def("__rmatmul__", &py_matmul)
        .def("relu", &py_relu)
        .def("exp", &py_exp)
        .def("log", &py_log)
        .def("abs", &py_abs)
        .def("sigmoid", &py_sigmoid)
        .def("tanh", &py_tanh)
        .def("sum", &py_sum)
        .def("mean", &py_mean)
        .def("shape", [](const Tensor& t) { return t.shape().dims; })
        .def("dtype", [](const Tensor& t) { return std::string(dtype_name(t.dtype())); })
        .def("numel", &Tensor::numel)
        .def("requires_grad", &Tensor::requires_grad)
        .def("requires_grad_", &Tensor::set_requires_grad)
        .def("zero_grad", &Tensor::zero_grad)
        .def("has_grad", &Tensor::has_grad)
        .def("init_grad", &Tensor::init_grad)
        .def("backward", &py_backward)
        .def("__repr__", &Tensor::repr)
        .def("tolist", [](const Tensor& t) {
            const float* p = static_cast<const float*>(t.data_ptr());
            int64_t n = t.numel();
            std::vector<float> v(p, p + n);
            return v;
        });
}
