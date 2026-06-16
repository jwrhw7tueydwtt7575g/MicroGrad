#include "micrograd/module.hpp"
#include <stdexcept>

namespace micrograd {

std::vector<Tensor*> Module::parameters() {
    std::vector<Tensor*> out;
    for (auto& t : own_params_) out.push_back(&t);
    for (auto& c : children_) {
        auto cp = c->parameters();
        for (auto* p : cp) out.push_back(p);
    }
    return out;
}

void Module::zero_grad() {
    for (auto* p : parameters()) {
        if (p->has_grad()) p->zero_grad();
    }
}

std::unordered_map<std::string, Tensor> Module::state_dict() const {
    return named_params_;
}

void Module::load_state_dict(const std::unordered_map<std::string, Tensor>& sd) {
    for (auto& kv : sd) {
        auto it = named_params_.find(kv.first);
        if (it == named_params_.end()) {
            throw std::runtime_error("load_state_dict: unknown param " + kv.first);
        }
        Tensor& dst = const_cast<Tensor&>(it->second);
        if (dst.shape() != kv.second.shape() || dst.dtype() != kv.second.dtype()) {
            throw std::runtime_error("load_state_dict: shape/dtype mismatch for " + kv.first);
        }
        int64_t N = dst.numel();
        std::memcpy(dst.data_ptr(), kv.second.data_ptr(), static_cast<size_t>(N) * dtype_size(dst.dtype()));
    }
}

}  // namespace micrograd
