#include "micrograd/softmax.hpp"
#include "micrograd/op_registry.hpp"
#include <cmath>
#include <algorithm>

namespace micrograd {

Tensor Softmax::forward(const Tensor& x) {
    int dim = dim_;
    int n = static_cast<int>(x.shape().rank());
    if (dim < 0) dim += n;
    const auto& dims = x.shape().dims;
    int64_t inner = 1;
    for (int i = dim + 1; i < n; ++i) inner *= dims[i];
    int64_t reduce = dims[dim];
    int64_t outer = x.numel() / (inner * reduce);

    const float* xp = static_cast<const float*>(x.data_ptr());
    Tensor y = Tensor::empty(x.shape(), x.dtype(), x.device());
    float* yp = static_cast<float*>(y.data_ptr());

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            // max along reduce axis
            float mx = -1e30f;
            for (int64_t r = 0; r < reduce; ++r) {
                float v = xp[o * reduce * inner + r * inner + i];
                if (v > mx) mx = v;
            }
            double s = 0.0;
            for (int64_t r = 0; r < reduce; ++r) {
                float e = std::exp(xp[o * reduce * inner + r * inner + i] - mx);
                yp[o * reduce * inner + r * inner + i] = e;
                s += e;
            }
            for (int64_t r = 0; r < reduce; ++r) {
                yp[o * reduce * inner + r * inner + i] /= static_cast<float>(s);
            }
        }
    }
    return y;
}

}  // namespace micrograd
