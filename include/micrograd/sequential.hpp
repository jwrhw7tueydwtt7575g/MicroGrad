// Sequential: list of layers.
#pragma once
#include "micrograd/module.hpp"
#include <vector>
#include <memory>

namespace micrograd {

class Sequential : public Module {
public:
    void add(std::shared_ptr<Module> m);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor*> parameters() override;
};

}  // namespace micrograd
