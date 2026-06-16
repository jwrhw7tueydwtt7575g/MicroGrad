#include "unary_kernels.hpp"
#include <cmath>

namespace micrograd {

void cpu_neg(const float* a, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = -a[i];
}

void cpu_exp(const float* a, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = std::exp(a[i]);
}

void cpu_log(const float* a, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = std::log(a[i]);
}

void cpu_abs(const float* a, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = std::fabs(a[i]);
}

void cpu_relu(const float* a, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

void cpu_sigmoid(const float* a, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = 1.0f / (1.0f + std::exp(-a[i]));
}

void cpu_tanh(const float* a, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = std::tanh(a[i]);
}

}  // namespace micrograd
