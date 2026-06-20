// Tensor: a handle to a Storage + metadata + an autograd anchor.
#pragma once
#include "micrograd/storage.hpp"
#include "micrograd/shape.hpp"
#include "micrograd/dtype.hpp"
#include "micrograd/device.hpp"
#include "micrograd/ir_fwd.hpp"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace micrograd {

class Tensor {
public:
    Tensor();
    Tensor(StorageHandle data, Shape shape, DType dtype, Device dev, bool requires_grad = false);

    // Metadata
    const Shape& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    int64_t numel() const { return shape_.numel(); }
    size_t nbytes() const { return static_cast<size_t>(numel()) * dtype_size(dtype_); }

    // Raw access
    void* data_ptr();
    const void* data_ptr() const;
    StorageHandle storage() const { return data_; }

    // Autograd
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool v);
    const Tensor& grad() const;
    Tensor& grad();
    const std::shared_ptr<Tensor>& grad_ptr() const { return grad_; }
    bool has_grad() const { return grad_ != nullptr; }
    void zero_grad();                  // fill grad_ with zeros
    void init_grad();                  // allocate grad_ if missing

    // Producer anchor (for autograd). producer_ is owned by the active Graph.
    // We store the Graph pointer + node id (not an IRNode*) because the
    // Graph's nodes vector can reallocate, which would invalidate raw
    // IRNode* references. To get the IRNode*, use graph()->node(node_id).
    struct ProducerRef { Graph* graph; int node_id; };
    ProducerRef producer() const { return producer_; }
    bool has_producer() const { return producer_.graph != nullptr; }
    void set_producer(Graph* g, int node_id) { producer_.graph = g; producer_.node_id = node_id; }
    void clear_producer() { producer_.graph = nullptr; producer_.node_id = -1; }
    IRNode* producer_node() const;  // resolves producer() to IRNode* (may be nullptr)

    // Helpers used by the kernel layer.
    static Tensor empty(Shape s, DType d = DType::F32, Device dev = Device::cpu());
    static Tensor full(Shape s, float v, DType d = DType::F32, Device dev = Device::cpu());
    static Tensor zeros(Shape s, DType d = DType::F32, Device dev = Device::cpu());
    static Tensor ones(Shape s, DType d = DType::F32, Device dev = Device::cpu());

    std::string repr() const;

private:
    StorageHandle data_;
    std::shared_ptr<class Tensor> grad_;     // shared so .grad() of leaf can be reused
    Shape shape_;
    DType dtype_;
    Device device_;
    bool requires_grad_;
    ProducerRef producer_{nullptr, -1};
};

}  // namespace micrograd
