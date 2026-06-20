// pybind11 tensor binding
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "micrograd/tensor.hpp"
#include "micrograd/op_registry.hpp"
#include "micrograd/ir.hpp"
#include "micrograd/autograd.hpp"
#include "micrograd/function.hpp"
#include "micrograd/active_fn.hpp"

namespace py = pybind11;
using namespace micrograd;

namespace {

Tensor py_add(const Tensor& a, const Tensor& b) { return run_op_ptrs("add", {&a, &b})[0]; }
Tensor py_sub(const Tensor& a, const Tensor& b) { return run_op_ptrs("sub", {&a, &b})[0]; }
Tensor py_mul(const Tensor& a, const Tensor& b) { return run_op_ptrs("mul", {&a, &b})[0]; }
Tensor py_div(const Tensor& a, const Tensor& b) { return run_op_ptrs("div", {&a, &b})[0]; }
Tensor py_pow(const Tensor& a, const Tensor& b) { return run_op_ptrs("pow", {&a, &b})[0]; }
Tensor py_matmul(const Tensor& a, const Tensor& b) { return run_op_ptrs("matmul", {&a, &b})[0]; }
Tensor py_neg(const Tensor& a) { return run_op_ptrs("neg", {&a})[0]; }
Tensor py_relu(const Tensor& a) { return run_op_ptrs("relu", {&a})[0]; }
Tensor py_exp(const Tensor& a) { return run_op_ptrs("exp", {&a})[0]; }
Tensor py_log(const Tensor& a) { return run_op_ptrs("log", {&a})[0]; }
Tensor py_abs(const Tensor& a) { return run_op_ptrs("abs", {&a})[0]; }
Tensor py_sigmoid(const Tensor& a) { return run_op_ptrs("sigmoid", {&a})[0]; }
Tensor py_tanh(const Tensor& a) { return run_op_ptrs("tanh", {&a})[0]; }
Tensor py_sum(const Tensor& a) { return run_op_ptrs("sum", {&a})[0]; }
Tensor py_mean(const Tensor& a) { return run_op_ptrs("mean", {&a})[0]; }
Tensor py_conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, const Tensor& stride, const Tensor& padding) {
    return run_op_ptrs("conv2d", {&x, &weight, &bias, &stride, &padding})[0];
}

// Scalar helpers: convert a Python numeric to a 0-d tensor with the same
// dtype/device as `ref`. Used by the scalar overloads of the operator
// bindings below.
Tensor scalar_to_tensor(const Tensor& ref, double v) {
    return Tensor::full(Shape{}, static_cast<float>(v), ref.dtype(), ref.device());
}

Tensor py_add_scalar(const Tensor& a, double b) { return py_add(a, scalar_to_tensor(a, b)); }
Tensor py_sub_scalar(const Tensor& a, double b) { return py_sub(a, scalar_to_tensor(a, b)); }
Tensor py_mul_scalar(const Tensor& a, double b) { return py_mul(a, scalar_to_tensor(a, b)); }
Tensor py_div_scalar(const Tensor& a, double b) { return py_div(a, scalar_to_tensor(a, b)); }
Tensor py_pow_scalar(const Tensor& a, double b) { return py_pow(a, scalar_to_tensor(a, b)); }
Tensor py_matmul_scalar(const Tensor& a, double b) {
    // matmul with a 0-d is not well-defined; raise instead of silently broadcasting.
    throw std::runtime_error("matmul: scalar operand not supported");
}
Tensor py_radd_scalar(const Tensor& a, double b) { return py_add(scalar_to_tensor(a, b), a); }
Tensor py_rsub_scalar(const Tensor& a, double b) { return py_sub(scalar_to_tensor(a, b), a); }
Tensor py_rmul_scalar(const Tensor& a, double b) { return py_mul(scalar_to_tensor(a, b), a); }
Tensor py_rdiv_scalar(const Tensor& a, double b) { return py_div(scalar_to_tensor(a, b), a); }
Tensor py_rmatmul_scalar(const Tensor& a, double b) { return py_matmul_scalar(a, b); }

void py_backward(Tensor& t) {
    // Standard path: the tracer has an active Function that owns the graph
    // (set up by Python's `with Function():` context, or by run_op's
    // transient-graph fallback).
    Tracer& tr = Tracer::current();
    if (tr.function) {
        tr.function->backward(t);
        if (active_function() == nullptr) {
            tr.graph = nullptr;
            tr.function = nullptr;
        }
        return;
    }
    // Fallback: locate the Function via the loss's producer IRNode's graph
    // owner. This handles `loss.backward()` called outside an explicit
    // `with Function():` block — the transient Function created by run_op
    // is kept alive by Graph::owner_keepalive.
    auto prod_ref = t.producer();
    if (!prod_ref.graph || !prod_ref.graph->owner) {
        // Nothing to backprop into.
        return;
    }
    prod_ref.graph->owner->backward(t);
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
        .def("__add__", &py_add_scalar)
        .def("__radd__", &py_add)
        .def("__radd__", &py_radd_scalar)
        .def("__sub__", &py_sub)
        .def("__sub__", &py_sub_scalar)
        .def("__rsub__", [](const Tensor& a, const Tensor& b) { return py_sub(b, a); })
        .def("__rsub__", &py_rsub_scalar)
        .def("__mul__", &py_mul)
        .def("__mul__", &py_mul_scalar)
        .def("__rmul__", &py_mul)
        .def("__rmul__", &py_rmul_scalar)
        .def("__truediv__", &py_div)
        .def("__truediv__", &py_div_scalar)
        .def("__rtruediv__", &py_rdiv_scalar)
        .def("__neg__", &py_neg)
        .def("__pow__", &py_pow)
        .def("__pow__", &py_pow_scalar)
        .def("__matmul__", &py_matmul)
        .def("__matmul__", &py_matmul_scalar)
        .def("__rmatmul__", &py_matmul)
        .def("__rmatmul__", &py_rmatmul_scalar)
        .def("relu", &py_relu)
        .def("exp", &py_exp)
        .def("log", &py_log)
        .def("abs", &py_abs)
        .def("sigmoid", &py_sigmoid)
        .def("tanh", &py_tanh)
        .def("sum", &py_sum)
        .def("mean", &py_mean)
        .def("conv2d", &py_conv2d)
        .def("shape", [](const Tensor& t) { return t.shape().dims; })
        .def("dtype", [](const Tensor& t) { return std::string(dtype_name(t.dtype())); })
        .def("numel", &Tensor::numel)
        .def("requires_grad", &Tensor::requires_grad)
        .def("requires_grad_", &Tensor::set_requires_grad)
        .def("zero_grad", &Tensor::zero_grad)
        .def("has_grad", &Tensor::has_grad)
        .def("has_producer", &Tensor::has_producer)
        .def("grad", [](Tensor& t) -> Tensor& { return t.grad(); })
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
