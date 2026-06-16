#include "micrograd/relu.hpp"
#include "micrograd/op_registry.hpp"

namespace micrograd {

Tensor ReLU::forward(const Tensor& x) {
    return run_op("relu", {x})[0];
}

}  // namespace micrograd
