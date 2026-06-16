// SGD: w -= lr * grad
#pragma once
#include "micrograd/optimizer.hpp"

namespace micrograd {

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> params, float lr, float momentum = 0.0f, float weight_decay = 0.0f);
    void step() override;
private:
    float momentum_;
    float weight_decay_;
    std::unordered_map<Tensor*, Tensor> velocity_;
};

}  // namespace micrograd
