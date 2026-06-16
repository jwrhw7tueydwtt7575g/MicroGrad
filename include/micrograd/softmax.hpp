// Softmax activation as a Module.
#pragma once
#include "micrograd/module.hpp"

namespace micrograd {

class Softmax : public Module {
public:
    explicit Softmax(int dim = -1) : dim_(dim) {}
    Tensor forward(const Tensor& x) override;
private:
    int dim_;
};

}  // namespace micrograd
