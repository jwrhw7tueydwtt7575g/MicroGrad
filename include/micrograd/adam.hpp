// Adam: per-parameter adaptive learning rate.
#pragma once
#include "micrograd/optimizer.hpp"

namespace micrograd {

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> params, float lr = 1e-3f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
         float weight_decay = 0.0f);
    void step() override;
    int step_count() const { return t_; }
private:
    float beta1_, beta2_, eps_, weight_decay_;
    int t_ = 0;
    std::unordered_map<Tensor*, Tensor> m_;     // first moment
    std::unordered_map<Tensor*, Tensor> v_;     // second moment
};

}  // namespace micrograd
