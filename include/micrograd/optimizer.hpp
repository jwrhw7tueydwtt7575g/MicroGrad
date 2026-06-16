// Optimizer: base class.
#pragma once
#include "micrograd/tensor.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

namespace micrograd {

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor*> params, float lr);
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad();

    float lr() const { return lr_; }
    void set_lr(float v) { lr_ = v; }
    const std::vector<Tensor*>& parameters() const { return params_; }

protected:
    std::vector<Tensor*> params_;
    float lr_;
};

}  // namespace micrograd
