#include "binary_kernels.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace micrograd {

std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b) {
    int an = static_cast<int>(a.size());
    int bn = static_cast<int>(b.size());
    int rn = std::max(an, bn);
    std::vector<int64_t> out(rn, 1);
    for (int i = 0; i < rn; ++i) {
        int64_t ad = (i < rn - an) ? 1 : a[i - (rn - an)];
        int64_t bd = (i < rn - bn) ? 1 : b[i - (rn - bn)];
        if (ad == bd) out[i] = ad;
        else if (ad == 1) out[i] = bd;
        else if (bd == 1) out[i] = ad;
        else throw std::runtime_error("broadcast_shape: incompatible");
    }
    return out;
}

std::vector<int64_t> row_strides(const std::vector<int64_t>& shape) {
    int n = static_cast<int>(shape.size());
    std::vector<int64_t> s(n, 1);
    for (int i = n - 2; i >= 0; --i) s[i] = s[i + 1] * shape[i + 1];
    return s;
}

void cpu_add(const float* a, const float* b, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

void cpu_sub(const float* a, const float* b, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
}

void cpu_mul(const float* a, const float* b, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
}

void cpu_div(const float* a, const float* b, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = a[i] / b[i];
}

void cpu_pow(const float* a, const float* b, float* c, int64_t n) {
    for (int64_t i = 0; i < n; ++i) c[i] = std::pow(a[i], b[i]);
}

}  // namespace micrograd
