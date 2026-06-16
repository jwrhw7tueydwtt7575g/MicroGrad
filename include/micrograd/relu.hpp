// ReLU activation as a Module.
#pragma once
#include "micrograd/module.hpp"

namespace micrograd {

class ReLU : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

}  // namespace micrograd
