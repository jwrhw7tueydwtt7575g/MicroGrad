// Op: the trait for a single op implementation.
#pragma once
#include "micrograd/tensor.hpp"
#include "micrograd/shape.hpp"
#include "micrograd/dtype.hpp"
#include "micrograd/device.hpp"
#include <functional>
#include <string>
#include <vector>
#include <memory>

namespace micrograd {

// Initialize all built-in CPU ops. Idempotent.
void init_cpu_ops();

// Sum-reduce `og` along axes that were broadcast to produce `target`'s
// shape; result is written into `dst`. Used by binary-op backward paths.
void reduce_to_input_shape(const Tensor& og, const Shape& target, Tensor& dst);

// Convenience alias.
using TensorVec = std::vector<Tensor>;

// Forward declarations
struct Op;

// CPU forward: takes input tensors, returns output tensors.
using ForwardFn = std::function<TensorVec(const TensorVec&)>;

// CPU backward: takes (outputs, output_grads, inputs, input_grads_inout) and
// accumulates into input_grads_inout. Backward runs once per op during the
// IR reverse pass.
using BackwardFn = std::function<void(const TensorVec& outputs,
                                      const TensorVec& output_grads,
                                      const TensorVec& inputs,
                                      TensorVec& input_grads)>;

using InferShapeFn = std::function<std::vector<Shape>(const std::vector<Shape>&)>;

struct Op {
    std::string name;
    DType dtype;
    Device device;
    ForwardFn forward;
    BackwardFn backward;
    InferShapeFn infer_shape;

    // Static (op-level) attrs: e.g. broadcast flags, axes.
    // The Python op_def path attaches extras via attr_bytes.
    std::vector<uint8_t> attr_bytes;
};

// Lookup key: (name, dtype, device).
struct OpKey {
    std::string name;
    DType dtype;
    Device device;
    bool operator==(const OpKey& o) const {
        return name == o.name && dtype == o.dtype && device == o.device;
    }
};

struct OpKeyHash {
    size_t operator()(const OpKey& k) const noexcept;
};

}  // namespace micrograd
