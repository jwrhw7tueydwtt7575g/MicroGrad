// Linear: y = x @ W^T + b
#pragma once
#include "micrograd/module.hpp"

namespace micrograd {

class Linear : public Module {
public:
    Linear(int in_features, int out_features, bool bias = true);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor*> parameters() override;

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }

private:
    int in_;
    int out_;
    bool has_bias_;
    Tensor weight_;          // (out, in)
    Tensor bias_;            // (out,)
};

}  // namespace micrograd
