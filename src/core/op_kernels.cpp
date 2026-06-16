// All CPU op registrations in one place.
#include "micrograd/op_registry.hpp"
#include "micrograd/tensor.hpp"
#include "binary_kernels.hpp"
#include "unary_kernels.hpp"
#include "reduce_kernels.hpp"
#include "matmul_kernels.hpp"
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace micrograd {

namespace {

// Helper: get the float* data of a tensor.
float* fdata(Tensor& t) { return static_cast<float*>(t.data_ptr()); }
const float* fdata(const Tensor& t) { return static_cast<const float*>(t.data_ptr()); }

// Helper: register a binary elementwise op with broadcasting.
void register_binary(const std::string& name, DType dt,
                     float (*scalar)(float, float),
                     void (*vec)(const float*, const float*, float*, int64_t)) {
    Op op;
    op.name = name;
    op.dtype = dt;
    op.device = Device::cpu();
    op.forward = [scalar, vec](const TensorVec& in) -> TensorVec {
        if (in.size() != 2) throw std::runtime_error(name + ": expected 2 inputs");
        const auto& a = in[0];
        const auto& b = in[1];
        auto out_shape = broadcast_shape(a.shape().dims, b.shape().dims);
        Tensor c = Tensor::empty(Shape(out_shape), a.dtype(), a.device());
        if (a.shape() == b.shape()) {
            vec(fdata(const_cast<Tensor&>(a)), fdata(const_cast<Tensor&>(b)),
                fdata(c), a.numel());
        } else {
            cpu_binary_broadcast(
                fdata(const_cast<Tensor&>(a)),
                fdata(const_cast<Tensor&>(b)),
                fdata(c),
                a.shape().dims, b.shape().dims, out_shape,
                [scalar](float x, float y) { return scalar(x, y); });
        }
        return {c};
    };
    op.backward = [scalar](const TensorVec& /*outputs*/,
                           const TensorVec& output_grads,
                           const TensorVec& inputs,
                           TensorVec& input_grads) {
        const Tensor& a = inputs[0];
        const Tensor& b = inputs[1];
        const Tensor& og = output_grads[0];
        // For v0.1, the generic backward for binary ops: d/da = scalar'(a,b) * og
        //                                         d/db = scalar'(a,b) * og (other chain)
        // We compute simple cases. For full correctness across all ops we
        // would need a per-op partial, but for the small set here it's clean.
        auto deriv_a = [scalar](float x, float y) {
            std::string n = "?";
            // We use a sentinel: in the closure above we know the name; pass
            // it via Op attrs. For simplicity in v0.1, the scalar() impl
            // is its own derivative via finite-diff is *not* what we want;
            // we explicitly dispatch on name.
            (void)x; (void)y; return 0.0f;
        };
        (void)deriv_a;  // unused placeholder; real impl below per-op
    };
    OpRegistry::instance().register_op(name, dt, std::move(op));
}

// We need per-op backward formulas. We register each binary op explicitly
// with its correct backward.

void register_add() {
    Op op; op.name = "add"; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [](const TensorVec& in) -> TensorVec {
        const auto& a = in[0]; const auto& b = in[1];
        auto sh = broadcast_shape(a.shape().dims, b.shape().dims);
        Tensor c = Tensor::empty(Shape(sh), a.dtype(), a.device());
        if (a.shape() == b.shape()) {
            cpu_add(fdata(const_cast<Tensor&>(a)), fdata(const_cast<Tensor&>(b)), fdata(c), a.numel());
        } else {
            cpu_binary_broadcast(fdata(const_cast<Tensor&>(a)),
                                 fdata(const_cast<Tensor&>(b)),
                                 fdata(c), a.shape().dims, b.shape().dims, sh,
                                 [](float x, float y) { return x + y; });
        }
        return {c};
    };
    op.backward = [](const TensorVec& /*outputs*/,
                     const TensorVec& output_grads,
                     const TensorVec& inputs,
                     TensorVec& input_grads) {
        // For add: d/da = og (broadcast-reduce if needed), d/db = og.
        const Tensor& a = inputs[0];
        const Tensor& b = inputs[1];
        const Tensor& og = output_grads[0];
        // Sum og along broadcast axes of a to get da, and along broadcast
        // axes of b to get db.
        reduce_to_input_shape(og, a.shape(), input_grads[0]);
        reduce_to_input_shape(og, b.shape(), input_grads[1]);
    };
    OpRegistry::instance().register_op("add", DType::F32, std::move(op));
}

void register_sub() {
    Op op; op.name = "sub"; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [](const TensorVec& in) -> TensorVec {
        const auto& a = in[0]; const auto& b = in[1];
        auto sh = broadcast_shape(a.shape().dims, b.shape().dims);
        Tensor c = Tensor::empty(Shape(sh), a.dtype(), a.device());
        if (a.shape() == b.shape()) {
            cpu_sub(fdata(const_cast<Tensor&>(a)), fdata(const_cast<Tensor&>(b)), fdata(c), a.numel());
        } else {
            cpu_binary_broadcast(fdata(const_cast<Tensor&>(a)),
                                 fdata(const_cast<Tensor&>(b)),
                                 fdata(c), a.shape().dims, b.shape().dims, sh,
                                 [](float x, float y) { return x - y; });
        }
        return {c};
    };
    op.backward = [](const TensorVec&, const TensorVec& output_grads,
                     const TensorVec& inputs, TensorVec& input_grads) {
        reduce_to_input_shape(output_grads[0], inputs[0].shape(), input_grads[0]);
        Tensor neg = Tensor::empty(input_grads[1].shape(), DType::F32, inputs[0].device());
        cpu_neg(fdata(output_grads[0]), fdata(neg), output_grads[0].numel());
        reduce_to_input_shape(neg, inputs[1].shape(), input_grads[1]);
    };
    OpRegistry::instance().register_op("sub", DType::F32, std::move(op));
}

void register_mul() {
    Op op; op.name = "mul"; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [](const TensorVec& in) -> TensorVec {
        const auto& a = in[0]; const auto& b = in[1];
        auto sh = broadcast_shape(a.shape().dims, b.shape().dims);
        Tensor c = Tensor::empty(Shape(sh), a.dtype(), a.device());
        if (a.shape() == b.shape()) {
            cpu_mul(fdata(const_cast<Tensor&>(a)), fdata(const_cast<Tensor&>(b)), fdata(c), a.numel());
        } else {
            cpu_binary_broadcast(fdata(const_cast<Tensor&>(a)),
                                 fdata(const_cast<Tensor&>(b)),
                                 fdata(c), a.shape().dims, b.shape().dims, sh,
                                 [](float x, float y) { return x * y; });
        }
        return {c};
    };
    op.backward = [](const TensorVec&, const TensorVec& output_grads,
                     const TensorVec& inputs, TensorVec& input_grads) {
        // d/da = b * og, d/db = a * og
        Tensor g_a = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        cpu_mul(fdata(const_cast<Tensor&>(inputs[1])), fdata(const_cast<Tensor&>(output_grads[0])),
                fdata(g_a), inputs[1].numel());
        reduce_to_input_shape(g_a, inputs[0].shape(), input_grads[0]);
        Tensor g_b = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        cpu_mul(fdata(const_cast<Tensor&>(inputs[0])), fdata(const_cast<Tensor&>(output_grads[0])),
                fdata(g_b), inputs[0].numel());
        reduce_to_input_shape(g_b, inputs[1].shape(), input_grads[1]);
    };
    OpRegistry::instance().register_op("mul", DType::F32, std::move(op));
}

void register_div() {
    Op op; op.name = "div"; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [](const TensorVec& in) -> TensorVec {
        const auto& a = in[0]; const auto& b = in[1];
        auto sh = broadcast_shape(a.shape().dims, b.shape().dims);
        Tensor c = Tensor::empty(Shape(sh), a.dtype(), a.device());
        if (a.shape() == b.shape()) {
            cpu_div(fdata(const_cast<Tensor&>(a)), fdata(const_cast<Tensor&>(b)), fdata(c), a.numel());
        } else {
            cpu_binary_broadcast(fdata(const_cast<Tensor&>(a)),
                                 fdata(const_cast<Tensor&>(b)),
                                 fdata(c), a.shape().dims, b.shape().dims, sh,
                                 [](float x, float y) { return x / y; });
        }
        return {c};
    };
    op.backward = [](const TensorVec&, const TensorVec& output_grads,
                     const TensorVec& inputs, TensorVec& input_grads) {
        // d/da = og / b; d/db = -og * a / b^2
        int64_t n = output_grads[0].numel();
        Tensor g_a(output_grads[0].shape(), DType::F32, output_grads[0].device());
        cpu_div(fdata(const_cast<Tensor&>(output_grads[0])),
                fdata(const_cast<Tensor&>(inputs[1])), fdata(g_a), n);
        reduce_to_input_shape(g_a, inputs[0].shape(), input_grads[0]);
        Tensor g_b(output_grads[0].shape(), DType::F32, output_grads[0].device());
        Tensor tmp1(output_grads[0].shape(), DType::F32, output_grads[0].device());
        Tensor tmp2(output_grads[0].shape(), DType::F32, output_grads[0].device());
        cpu_mul(fdata(const_cast<Tensor&>(output_grads[0])),
                fdata(const_cast<Tensor&>(inputs[0])), fdata(tmp1), n);
        cpu_mul(fdata(const_cast<Tensor&>(inputs[1])),
                fdata(const_cast<Tensor&>(inputs[1])), fdata(tmp2), n);
        cpu_div(fdata(tmp1), fdata(tmp2), fdata(g_b), n);
        Tensor neg_gb(g_b.shape(), DType::F32, g_b.device());
        cpu_neg(fdata(g_b), fdata(neg_gb), g_b.numel());
        reduce_to_input_shape(neg_gb, inputs[1].shape(), input_grads[1]);
    };
    OpRegistry::instance().register_op("div", DType::F32, std::move(op));
}

void register_pow() {
    Op op; op.name = "pow"; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [](const TensorVec& in) -> TensorVec {
        const auto& a = in[0]; const auto& b = in[1];
        auto sh = broadcast_shape(a.shape().dims, b.shape().dims);
        Tensor c = Tensor::empty(Shape(sh), a.dtype(), a.device());
        if (a.shape() == b.shape()) {
            cpu_pow(fdata(const_cast<Tensor&>(a)), fdata(const_cast<Tensor&>(b)), fdata(c), a.numel());
        } else {
            cpu_binary_broadcast(fdata(const_cast<Tensor&>(a)),
                                 fdata(const_cast<Tensor&>(b)),
                                 fdata(c), a.shape().dims, b.shape().dims, sh,
                                 [](float x, float y) { return std::pow(x, y); });
        }
        return {c};
    };
    op.backward = [](const TensorVec&, const TensorVec& output_grads,
                     const TensorVec& inputs, TensorVec& input_grads) {
        int64_t n = output_grads[0].numel();
        // d/da = b * a^(b-1) * og
        Tensor g_a(output_grads[0].shape(), DType::F32, output_grads[0].device());
        Tensor b_minus_one(inputs[1].shape(), DType::F32, inputs[1].device());
        // We compute b - 1 using a tmp tensor.
        Tensor ones = Tensor::ones(inputs[1].shape(), DType::F32, inputs[1].device());
        cpu_sub(fdata(const_cast<Tensor&>(inputs[1])), fdata(ones), fdata(b_minus_one), inputs[1].numel());
        Tensor a_pow(output_grads[0].shape(), DType::F32, output_grads[0].device());
        cpu_pow(fdata(const_cast<Tensor&>(inputs[0])), fdata(b_minus_one), fdata(a_pow), n);
        cpu_mul(fdata(a_pow), fdata(const_cast<Tensor&>(inputs[1])), fdata(g_a), n);
        cpu_mul(fdata(g_a), fdata(const_cast<Tensor&>(output_grads[0])), fdata(g_a), n);
        reduce_to_input_shape(g_a, inputs[0].shape(), input_grads[0]);
        // d/db = a^b * log(a) * og
        Tensor g_b(output_grads[0].shape(), DType::F32, output_grads[0].device());
        Tensor ab(n == 0 ? Shape{0} : inputs[0].shape(), DType::F32, inputs[0].device());
        Tensor a_safe(inputs[0].shape(), DType::F32, inputs[0].device());
        // Clamp a to >0 for log.
        int64_t na = inputs[0].numel();
        const float* ap = fdata(const_cast<Tensor&>(inputs[0]));
        float* a_safe_p = fdata(a_safe);
        for (int64_t i = 0; i < na; ++i) a_safe_p[i] = std::max(ap[i], 1e-12f);
        cpu_pow(fdata(a_safe), fdata(const_cast<Tensor&>(inputs[1])), fdata(ab), n);
        Tensor log_a(inputs[0].shape(), DType::F32, inputs[0].device());
        cpu_log(fdata(a_safe), fdata(log_a), na);
        cpu_mul(fdata(ab), fdata(log_a), fdata(g_b), n);
        cpu_mul(fdata(g_b), fdata(const_cast<Tensor&>(output_grads[0])), fdata(g_b), n);
        reduce_to_input_shape(g_b, inputs[1].shape(), input_grads[1]);
    };
    OpRegistry::instance().register_op("pow", DType::F32, std::move(op));
}

// Unary ops
template <typename Fwd, typename Bwd>
void register_unary(const std::string& name, Fwd fwd, Bwd bwd) {
    Op op; op.name = name; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [fwd](const TensorVec& in) -> TensorVec {
        const auto& a = in[0];
        Tensor c = Tensor::empty(a.shape(), a.dtype(), a.device());
        fwd(fdata(const_cast<Tensor&>(a)), fdata(c), a.numel());
        return {c};
    };
    op.backward = [bwd](const TensorVec& /*outputs*/,
                        const TensorVec& output_grads,
                        const TensorVec& inputs,
                        TensorVec& input_grads) {
        const Tensor& a = inputs[0];
        const Tensor& og = output_grads[0];
        Tensor g = Tensor::empty(a.shape(), a.dtype(), a.device());
        bwd(fdata(const_cast<Tensor&>(a)), fdata(og), fdata(g), a.numel());
        // in-place add into input_grads[0]
        int64_t n = a.numel();
        float* dp = fdata(input_grads[0]);
        const float* sp = fdata(g);
        for (int64_t i = 0; i < n; ++i) dp[i] += sp[i];
    };
    OpRegistry::instance().register_op(name, DType::F32, std::move(op));
}

void register_unary_ops() {
    register_unary("neg", cpu_neg, [](const float*, const float* og, float* g, int64_t n) {
        for (int64_t i = 0; i < n; ++i) g[i] = -og[i];
    });
    register_unary("exp", cpu_exp, [](const float* a, const float* og, float* g, int64_t n) {
        for (int64_t i = 0; i < n; ++i) g[i] = std::exp(a[i]) * og[i];
    });
    register_unary("log", cpu_log, [](const float* a, const float* og, float* g, int64_t n) {
        for (int64_t i = 0; i < n; ++i) g[i] = og[i] / a[i];
    });
    register_unary("abs", cpu_abs, [](const float* a, const float* og, float* g, int64_t n) {
        for (int64_t i = 0; i < n; ++i) g[i] = (a[i] > 0 ? 1.0f : (a[i] < 0 ? -1.0f : 0.0f)) * og[i];
    });
    register_unary("relu", cpu_relu, [](const float* a, const float* og, float* g, int64_t n) {
        for (int64_t i = 0; i < n; ++i) g[i] = (a[i] > 0 ? og[i] : 0.0f);
    });
    register_unary("sigmoid", cpu_sigmoid, [](const float* a, const float* og, float* g, int64_t n) {
        for (int64_t i = 0; i < n; ++i) {
            float s = 1.0f / (1.0f + std::exp(-a[i]));
            g[i] = s * (1.0f - s) * og[i];
        }
    });
    register_unary("tanh", cpu_tanh, [](const float* a, const float* og, float* g, int64_t n) {
        for (int64_t i = 0; i < n; ++i) {
            float t = std::tanh(a[i]);
            g[i] = (1.0f - t * t) * og[i];
        }
    });
}

// Reductions
void register_sum() {
    Op op; op.name = "sum"; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [](const TensorVec& in) -> TensorVec {
        const auto& a = in[0];
        // attrs: int32 axis (0xFFFFFFFF means "all")
        int axis = -1;  // default: all
        Tensor c = Tensor::empty(Shape{}, a.dtype(), a.device());
        cpu_sum_all(fdata(const_cast<Tensor&>(a)), fdata(c), a.numel());
        (void)axis;
        return {c};
    };
    op.backward = [](const TensorVec&, const TensorVec& output_grads,
                     const TensorVec& inputs, TensorVec& input_grads) {
        // og is a scalar; broadcast to input shape.
        const Tensor& a = inputs[0];
        const float og = *fdata(const_cast<Tensor&>(output_grads[0]));
        float* g = fdata(input_grads[0]);
        for (int64_t i = 0; i < a.numel(); ++i) g[i] += og;
    };
    OpRegistry::instance().register_op("sum", DType::F32, std::move(op));
}

void register_mean() {
    Op op; op.name = "mean"; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [](const TensorVec& in) -> TensorVec {
        const auto& a = in[0];
        Tensor c = Tensor::empty(Shape{}, a.dtype(), a.device());
        cpu_mean_all(fdata(const_cast<Tensor&>(a)), fdata(c), a.numel());
        return {c};
    };
    op.backward = [](const TensorVec&, const TensorVec& output_grads,
                     const TensorVec& inputs, TensorVec& input_grads) {
        const Tensor& a = inputs[0];
        const float og = *fdata(const_cast<Tensor&>(output_grads[0]));
        float scale = og / static_cast<float>(a.numel());
        float* g = fdata(input_grads[0]);
        for (int64_t i = 0; i < a.numel(); ++i) g[i] += scale;
    };
    OpRegistry::instance().register_op("mean", DType::F32, std::move(op));
}

void register_matmul() {
    Op op; op.name = "matmul"; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [](const TensorVec& in) -> TensorVec {
        const auto& a = in[0]; const auto& b = in[1];
        if (a.shape().rank() != 2 || b.shape().rank() != 2) {
            throw std::runtime_error("matmul: v0.1 only supports 2D @ 2D");
        }
        int M = static_cast<int>(a.shape().dims[0]);
        int K = static_cast<int>(a.shape().dims[1]);
        int Kb = static_cast<int>(b.shape().dims[0]);
        int N = static_cast<int>(b.shape().dims[1]);
        if (K != Kb) throw std::runtime_error("matmul: shape mismatch");
        Tensor c = Tensor::empty(Shape({M, N}), a.dtype(), a.device());
        cpu_matmul(fdata(const_cast<Tensor&>(a)), fdata(const_cast<Tensor&>(b)),
                   fdata(c), M, K, N);
        return {c};
    };
    op.backward = [](const TensorVec&, const TensorVec& output_grads,
                     const TensorVec& inputs, TensorVec& input_grads) {
        const Tensor& a = inputs[0]; const Tensor& b = inputs[1];
        int M = static_cast<int>(a.shape().dims[0]);
        int K = static_cast<int>(a.shape().dims[1]);
        int N = static_cast<int>(b.shape().dims[1]);
        // dA = og @ B^T
        cpu_matmul_T(fdata(const_cast<Tensor&>(output_grads[0])),
                     fdata(const_cast<Tensor&>(b)),
                     fdata(input_grads[0]), M, N, K);
        // dB = A^T @ og
        Tensor at(K, M, a.dtype(), a.device());
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) at.data_ptr<float>()[k * M + i] = a.data_ptr<float>()[i * K + k];
        cpu_matmul(fdata(at), fdata(const_cast<Tensor&>(output_grads[0])),
                   fdata(input_grads[1]), K, M, N);
    };
    OpRegistry::instance().register_op("matmul", DType::F32, std::move(op));
}

}  // namespace

// Public init function called from the C++ runtime once at module load.
void init_cpu_ops() {
    register_add();
    register_sub();
    register_mul();
    register_div();
    register_pow();
    register_unary_ops();
    register_sum();
    register_mean();
    register_matmul();
}

// Helper used by broadcast-reduce backward paths.
void reduce_to_input_shape(const Tensor& og, const Shape& target, Tensor& dst) {
    int64_t n = og.numel();
    int64_t tn = target.numel();
    if (n == tn) {
        std::memcpy(dst.data_ptr(), og.data_ptr(), static_cast<size_t>(n) * sizeof(float));
        return;
    }
    // Sum-reduce og along axes that were broadcast.
    // Build strides for og in C-order.
    auto og_strides = row_strides(og.shape().dims);
    auto dst_strides = row_strides(target.dims);
    // For each element of dst, find the corresponding og indices.
    int rn = static_cast<int>(og.shape().rank());
    int tn_rank = static_cast<int>(target.rank());
    // Compute which axes of og are "kept" vs "reduced" (size 1).
    std::vector<int> og_axes(rn);
    for (int i = 0; i < rn; ++i) og_axes[i] = i - (rn - tn_rank);
    std::memset(dst.data_ptr(), 0, static_cast<size_t>(tn) * sizeof(float));
    for (int64_t idx = 0; idx < n; ++idx) {
        // Decode og linear index into coord.
        std::vector<int64_t> coord(rn);
        int64_t tmp = idx;
        for (int i = 0; i < rn; ++i) {
            coord[i] = tmp / og_strides[i];
            tmp %= og_strides[i];
        }
        // Project to target.
        int64_t dst_idx = 0;
        for (int i = 0; i < rn; ++i) {
            int ti = og_axes[i];
            int64_t c;
            if (ti < 0) c = 0;
            else if (target.dims[ti] == 1) c = 0;
            else c = coord[i];
            if (ti >= 0) dst_idx += c * dst_strides[ti];
        }
        dst.data_ptr<float>()[dst_idx] += og.data_ptr<float>()[idx];
    }
}

}  // namespace micrograd
