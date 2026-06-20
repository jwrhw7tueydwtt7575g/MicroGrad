#include "micrograd/linear.hpp"
#include "micrograd/op_registry.hpp"
#include "micrograd/op.hpp"
#include <random>
#include <cmath>

namespace micrograd {

Linear::Linear(int in_features, int out_features, bool bias)
    : in_(in_features), out_(out_features), has_bias_(bias) {
    // Kaiming-uniform init with unique seed per layer.
    static thread_local uint64_t seed_counter = 0;
    float bound = std::sqrt(1.0f / static_cast<float>(in_features));
    std::mt19937 rng(42 + seed_counter++);
    std::uniform_real_distribution<float> d(-bound, bound);
    // Store weight as (in, out) so y = x @ W + b is a plain matmul.
    weight_ = Tensor::empty(Shape({in_features, out_features}), DType::F32, Device::cpu());
    weight_.set_requires_grad(true);
    auto* wp = static_cast<float*>(weight_.data_ptr());
    for (int64_t i = 0; i < weight_.numel(); ++i) wp[i] = d(rng);
    named_params_["weight"] = weight_;
    own_params_.push_back(weight_);
    if (has_bias_) {
        bias_ = Tensor::zeros(Shape({out_features}), DType::F32, Device::cpu());
        bias_.set_requires_grad(true);
        named_params_["bias"] = bias_;
        own_params_.push_back(bias_);
    }
}

Tensor Linear::forward(const Tensor& x) {
    // x: (..., in); weight: (in, out); bias: (out,)
    // y = x @ weight + bias
    Tensor y = run_op("matmul", {x, weight_})[0];
    if (has_bias_) {
        y = run_op("add", {y, bias_})[0];
    }
    return y;
}

std::vector<Tensor*> Linear::parameters() {
    std::vector<Tensor*> out;
    out.push_back(&weight_);
    if (has_bias_) out.push_back(&bias_);
    return out;
}

}  // namespace micrograd
