#include "micrograd/tensor.hpp"
#include "micrograd/ir.hpp"
#include "micrograd/storage.hpp"
#include "micrograd/stream.hpp"
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace micrograd {

IRNode* Tensor::producer_node() const {
    if (!producer_.graph) return nullptr;
    if (producer_.node_id < 0) return nullptr;
    if (producer_.node_id >= static_cast<int>(producer_.graph->nodes.size())) return nullptr;
    return &producer_.graph->node(producer_.node_id);
}

Tensor::Tensor()
    : shape_(Shape({0})),
      dtype_(DType::F32),
      device_(Device::cpu()),
      requires_grad_(false) {}

Tensor::Tensor(StorageHandle data, Shape shape, DType dtype, Device dev, bool requires_grad)
    : data_(std::move(data)),
      shape_(std::move(shape)),
      dtype_(dtype),
      device_(dev),
      requires_grad_(requires_grad) {}

void* Tensor::data_ptr() {
    if (!data_) throw std::runtime_error("Tensor::data_ptr: no storage");
    return data_->raw();
}

const void* Tensor::data_ptr() const {
    if (!data_) throw std::runtime_error("Tensor::data_ptr: no storage");
    return data_->raw();
}

void Tensor::set_requires_grad(bool v) {
    requires_grad_ = v;
    if (!v) grad_.reset();
}

const Tensor& Tensor::grad() const {
    if (!grad_) throw std::runtime_error("Tensor::grad: no grad allocated");
    return *grad_;
}

Tensor& Tensor::grad() {
    if (!grad_) throw std::runtime_error("Tensor::grad: no grad allocated");
    return *grad_;
}

void Tensor::init_grad() {
    if (grad_) return;
    StorageHandle g = make_storage(device_, nbytes());
    grad_ = std::make_shared<Tensor>(g, shape_, dtype_, device_, false);
}

void Tensor::zero_grad() {
    if (!grad_) return;
    auto* p = static_cast<uint8_t*>(grad_->data_ptr());
    std::memset(p, 0, grad_->nbytes());
}

Tensor Tensor::empty(Shape s, DType d, Device dev) {
    int64_t n = s.numel();
    auto storage = make_storage(dev, static_cast<size_t>(n) * dtype_size(d));
    return Tensor(storage, std::move(s), d, dev);
}

Tensor Tensor::full(Shape s, float v, DType d, Device dev) {
    Tensor t = empty(std::move(s), d, dev);
    if (d == DType::F32) {
        auto* p = static_cast<float*>(t.data_ptr());
        int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) p[i] = v;
    } else {
        throw std::runtime_error("Tensor::full: dtype not supported");
    }
    return t;
}

Tensor Tensor::zeros(Shape s, DType d, Device dev) { return full(std::move(s), 0.0f, d, dev); }
Tensor Tensor::ones(Shape s, DType d, Device dev)  { return full(std::move(s), 1.0f, d, dev); }

std::string Tensor::repr() const {
    std::ostringstream s;
    s << "Tensor(shape=" << shape_.to_string()
      << ", dtype=" << dtype_name(dtype_)
      << ", device=" << device_.to_string()
      << ", requires_grad=" << (requires_grad_ ? "True" : "False") << ")";
    return s.str();
}

}  // namespace micrograd
