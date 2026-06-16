#include "micrograd/sgd.hpp"

namespace micrograd {

SGD::SGD(std::vector<Tensor*> params, float lr, float momentum, float weight_decay)
    : Optimizer(std::move(params), lr), momentum_(momentum), weight_decay_(weight_decay) {}

void SGD::step() {
    for (auto* p : params_) {
        if (!p->has_grad()) continue;
        int64_t n = p->numel();
        float* w = static_cast<float*>(p->data_ptr());
        const float* g = static_cast<const float*>(p->grad().data_ptr());
        if (momentum_ > 0.0f) {
            auto& v = velocity_[p];
            if (!v.has_grad() && v.numel() == 0) {
                v = Tensor::zeros(p->shape(), p->dtype(), p->device());
            }
            float* vp = static_cast<float*>(v.data_ptr());
            for (int64_t i = 0; i < n; ++i) {
                float gi = g[i] + weight_decay_ * w[i];
                vp[i] = momentum_ * vp[i] + gi;
                w[i] -= lr_ * vp[i];
            }
        } else {
            for (int64_t i = 0; i < n; ++i) {
                w[i] -= lr_ * (g[i] + weight_decay_ * w[i]);
            }
        }
    }
}

}  // namespace micrograd
