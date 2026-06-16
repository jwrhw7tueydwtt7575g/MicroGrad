// Module: base class. Each derived layer registers its parameters and
// implements forward().
#pragma once
#include "micrograd/tensor.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>

namespace micrograd {

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::vector<Tensor*> parameters();

    void zero_grad();
    std::unordered_map<std::string, Tensor> state_dict() const;
    void load_state_dict(const std::unordered_map<std::string, Tensor>& sd);

    // Walk the children list (used by Sequential, by parameters()).
    void add_child(const std::shared_ptr<Module>& m) { children_.push_back(m); }
    const std::vector<std::shared_ptr<Module>>& children() const { return children_; }

protected:
    std::vector<std::shared_ptr<Module>> children_;
    std::vector<Tensor> own_params_;          // direct parameters
    std::unordered_map<std::string, Tensor> named_params_;
};

}  // namespace micrograd
