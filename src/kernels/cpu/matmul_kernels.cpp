#include "matmul_kernels.hpp"
#include <vector>

namespace micrograd {

void cpu_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += static_cast<double>(A[i * K + k]) * B[k * N + j];
            C[i * N + j] = static_cast<float>(s);
        }
    }
}

void cpu_matmul_T(const float* A, const float* B, float* C, int M, int K, int N) {
    // B is (N, K) row-major; we treat it as B^T of shape (K, N).
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += static_cast<double>(A[i * K + k]) * B[j * K + k];
            C[i * N + j] = static_cast<float>(s);
        }
    }
}

}  // namespace micrograd
