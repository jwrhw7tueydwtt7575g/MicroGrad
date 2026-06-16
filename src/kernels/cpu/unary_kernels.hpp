// CPU unary elementwise kernels.
#pragma once
#include <cstdint>

namespace micrograd {

void cpu_neg(const float* a, float* c, int64_t n);
void cpu_exp(const float* a, float* c, int64_t n);
void cpu_log(const float* a, float* c, int64_t n);
void cpu_abs(const float* a, float* c, int64_t n);
void cpu_relu(const float* a, float* c, int64_t n);
void cpu_sigmoid(const float* a, float* c, int64_t n);
void cpu_tanh(const float* a, float* c, int64_t n);

}  // namespace micrograd
