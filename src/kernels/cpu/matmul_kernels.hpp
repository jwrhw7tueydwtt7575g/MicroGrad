// CPU matmul: 2D-only in v0.1.
#pragma once
#include <cstdint>

namespace micrograd {

// C[M,N] = A[M,K] @ B[K,N]
void cpu_matmul(const float* A, const float* B, float* C, int M, int K, int N);
// B^T: C[M,N] = A[M,K] @ B[N,K]^T
void cpu_matmul_T(const float* A, const float* B, float* C, int M, int K, int N);

}  // namespace micrograd
