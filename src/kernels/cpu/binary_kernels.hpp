// CPU binary elementwise kernels with broadcasting.
#pragma once
#include <cstdint>
#include <vector>

namespace micrograd {

// Broadcasting rules: align right; each dim is broadcast if either side is 1
// or they match. Caller passes already-broadcast logical dims; this function
// produces the broadcast output shape.
std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b);

// Compute total strides (in elements) for a shape, row-major.
std::vector<int64_t> row_strides(const std::vector<int64_t>& shape);

// Element-wise binary op with broadcasting. `op` takes (lhs, rhs) and
// returns the result. Both inputs and output are float32.
template <typename Op>
void cpu_binary_broadcast(const float* a, const float* b, float* c,
                          const std::vector<int64_t>& a_shape,
                          const std::vector<int64_t>& b_shape,
                          const std::vector<int64_t>& c_shape,
                          Op op) {
    int64_t n = 1;
    for (int64_t d : c_shape) n *= d;
    // Strides for A and B in C's shape (0 if broadcast).
    int rn = static_cast<int>(c_shape.size());
    int an = static_cast<int>(a_shape.size());
    int bn = static_cast<int>(b_shape.size());
    std::vector<int64_t> a_strides(rn, 0);
    std::vector<int64_t> b_strides(rn, 0);
    for (int i = 0; i < rn; ++i) {
        int ai = i - (rn - an);
        int bi = i - (rn - bn);
        int64_t a_dim = (ai >= 0) ? a_shape[ai] : 1;
        int64_t b_dim = (bi >= 0) ? b_shape[bi] : 1;
        if (a_dim == c_shape[i] && a_dim > 1) {
            int64_t s = 1;
            for (int j = ai + 1; j < an; ++j) s *= a_shape[j];
            a_strides[i] = s;
        }
        if (b_dim == c_shape[i] && b_dim > 1) {
            int64_t s = 1;
            for (int j = bi + 1; j < bn; ++j) s *= b_shape[j];
            b_strides[i] = s;
        }
    }
    std::vector<int64_t> c_strides = row_strides(c_shape);
    for (int64_t idx = 0; idx < n; ++idx) {
        int64_t off_a = 0, off_b = 0;
        int64_t tmp = idx;
        for (int i = 0; i < rn; ++i) {
            int64_t coord = tmp / c_strides[i];
            tmp %= c_strides[i];
            off_a += coord * a_strides[i];
            off_b += coord * b_strides[i];
        }
        c[idx] = op(a[off_a], b[off_b]);
    }
}

void cpu_add(const float* a, const float* b, float* c, int64_t n);
void cpu_sub(const float* a, const float* b, float* c, int64_t n);
void cpu_mul(const float* a, const float* b, float* c, int64_t n);
void cpu_div(const float* a, const float* b, float* c, int64_t n);
void cpu_pow(const float* a, const float* b, float* c, int64_t n);

}  // namespace micrograd
