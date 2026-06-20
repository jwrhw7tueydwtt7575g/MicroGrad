#include "micrograd/sequential.hpp"

namespace micrograd {

void Sequential::add(std::shared_ptr<Module> m) {
    children_.push_back(m);
}

Tensor Sequential::forward(const Tensor& x) {
    Tensor cur = x;
    for (auto& m : children_) {
        cur = m->forward(cur);
    }
    return cur;
}

std::vector<Tensor*> Sequential::parameters() {
    return Module::parameters();
}

}  // namespace micrograd
