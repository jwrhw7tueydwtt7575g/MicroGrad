// Autograd engine: backward implementation.
#pragma once
#include "micrograd/function.hpp"
#include "micrograd/tensor.hpp"

namespace micrograd {

// Backward entry point. Computes gradients of all leaf parameters with
// requires_grad=true. Single linear pass over the graph's reverse order
// with topological release.
void backward(Function& f, const Tensor& loss);

// Python-side no_grad() context manager helper.
class GradMode {
public:
    explicit GradMode(bool enabled) : prev(Tracer::current().recording) {
        Tracer::current().recording = enabled;
    }
    ~GradMode() { Tracer::current().recording = prev; }
    GradMode(const GradMode&) = delete;
    GradMode& operator=(const GradMode&) = delete;
private:
    bool prev;
};

inline GradMode no_grad() { return GradMode(false); }
inline GradMode enable_grad() { return GradMode(true); }

}  // namespace micrograd
