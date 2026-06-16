#include "micrograd/sequential.hpp"

namespace micrograd {

void Sequential::add(std::shared_ptr<Module> m) {
    children_.push_back(m);
    for (auto* p : m->parameters()) {
        own_params_.push_back(*p);
    }
}

Tensor Sequential::forward(const Tensor& x) {
    Tensor cur = x;
    for (auto& m : children_) {
        cur = m->forward(cur);
    }
    return cur;
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> out;
    for (auto* p : own_params_) out.push_back(&p);
    return out;
}

}  // namespace micrograd
