#include "micrograd/adam.hpp"
#include <cmath>

namespace micrograd {

Adam::Adam(std::vector<Tensor*> params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(std::move(params), lr),
      beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {}

void Adam::step() {
    t_++;
    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));
    for (auto* p : params_) {
        if (!p->has_grad()) continue;
        int64_t n = p->numel();
        float* w = static_cast<float*>(p->data_ptr());
        const float* g = static_cast<const float*>(p->grad().data_ptr());
        auto& m_t = m_[p];
        auto& v_t = v_[p];
        if (m_t.numel() == 0) {
            m_t = Tensor::zeros(p->shape(), p->dtype(), p->device());
            v_t = Tensor::zeros(p->shape(), p->dtype(), p->device());
        }
        float* mp = static_cast<float*>(m_t.data_ptr());
        float* vp = static_cast<float*>(v_t.data_ptr());
        for (int64_t i = 0; i < n; ++i) {
            float gi = g[i] + weight_decay_ * w[i];
            mp[i] = beta1_ * mp[i] + (1.0f - beta1_) * gi;
            vp[i] = beta2_ * vp[i] + (1.0f - beta2_) * gi * gi;
            float m_hat = mp[i] / bc1;
            float v_hat = vp[i] / bc2;
            w[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

}  // namespace micrograd
