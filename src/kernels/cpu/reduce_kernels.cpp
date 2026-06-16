#include "reduce_kernels.hpp"
#include <stdexcept>
#include <numeric>
#include "binary_kernels.hpp"

namespace micrograd {

void cpu_sum_all(const float* a, float* c, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += a[i];
    *c = static_cast<float>(s);
}

void cpu_mean_all(const float* a, float* c, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += a[i];
    *c = static_cast<float>(s / static_cast<double>(n));
}

void cpu_sum_axis(const float* a, float* c,
                  const std::vector<int64_t>& shape, int axis) {
    int n = static_cast<int>(shape.size());
    if (axis < 0) axis += n;
    if (axis < 0 || axis >= n) throw std::runtime_error("cpu_sum_axis: axis out of range");
    auto strides = row_strides(shape);
    int64_t outer = 1, reduce = shape[axis], inner = 1;
    for (int i = 0; i < axis; ++i) outer *= shape[i];
    for (int i = axis + 1; i < n; ++i) inner *= shape[i];
    int64_t out_idx = 0;
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            double s = 0.0;
            for (int64_t r = 0; r < reduce; ++r) {
                int64_t idx = o * reduce * inner + r * inner + i;
                s += a[idx];
            }
            c[out_idx++] = static_cast<float>(s);
        }
    }
}

void cpu_mean_axis(const float* a, float* c,
                   const std::vector<int64_t>& shape, int axis) {
    int n = static_cast<int>(shape.size());
    if (axis < 0) axis += n;
    auto strides = row_strides(shape);
    int64_t outer = 1, reduce = shape[axis], inner = 1;
    for (int i = 0; i < axis; ++i) outer *= shape[i];
    for (int i = axis + 1; i < n; ++i) inner *= shape[i];
    int64_t out_idx = 0;
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            double s = 0.0;
            for (int64_t r = 0; r < reduce; ++r) {
                int64_t idx = o * reduce * inner + r * inner + i;
                s += a[idx];
            }
            c[out_idx++] = static_cast<float>(s / static_cast<double>(reduce));
        }
    }
}

}  // namespace micrograd
