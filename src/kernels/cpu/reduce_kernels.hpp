// CPU reduction kernels: sum, mean over all or along a single axis.
#pragma once
#include <cstdint>
#include <vector>

namespace micrograd {

// Reduce all axes to scalar.
void cpu_sum_all(const float* a, float* c, int64_t n);
void cpu_mean_all(const float* a, float* c, int64_t n);
// Reduce along `axis` (axis can be negative). Output has rank-1 dims.
void cpu_sum_axis(const float* a, float* c,
                  const std::vector<int64_t>& shape, int axis);
void cpu_mean_axis(const float* a, float* c,
                   const std::vector<int64_t>& shape, int axis);

}  // namespace micrograd
