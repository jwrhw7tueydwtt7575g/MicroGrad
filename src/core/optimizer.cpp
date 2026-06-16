#include "micrograd/optimizer.hpp"
#include <stdexcept>

namespace micrograd {

Optimizer::Optimizer(std::vector<Tensor*> params, float lr)
    : params_(std::move(params)), lr_(lr) {}

void Optimizer::zero_grad() {
    for (auto* p : params_) {
        if (p->has_grad()) p->zero_grad();
    }
}

}  // namespace micrograd
