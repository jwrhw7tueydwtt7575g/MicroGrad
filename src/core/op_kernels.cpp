// All CPU op registrations in one place.
#include "micrograd/op_registry.hpp"
#include "micrograd/tensor.hpp"
#include "kernels/cpu/binary_kernels.hpp"
#include "kernels/cpu/unary_kernels.hpp"
#include "kernels/cpu/reduce_kernels.hpp"
#include "kernels/cpu/matmul_kernels.hpp"
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace micrograd {

namespace {

// Helper: get the float* data of a tensor.
float* fdata(Tensor& t) { return static_cast<float*>(t.data_ptr()); }
const float* fdata(const Tensor& t) { return static_cast<const float*>(t.data_ptr()); }

// Helper: run a binary op on two tensors, supporting broadcasting if shapes differ.
template <typename Op>
void run_binary_bwd_op(const Tensor& a, const Tensor& b, Tensor& c, Op op) {
    if (a.shape() == b.shape()) {
        int64_t n = a.numel();
        const float* ap = fdata(const_cast<Tensor&>(a));
        const float* bp = fdata(const_cast<Tensor&>(b));
        float* cp = fdata(c);
        for (int64_t i = 0; i < n; ++i) cp[i] = op(ap[i], bp[i]);
    } else {
        cpu_binary_broadcast(fdata(const_cast<Tensor&>(a)), fdata(const_cast<Tensor&>(b)), fdata(c),
                             a.shape().dims, b.shape().dims, c.shape().dims, op);
    }
}

// Helper: register a binary elementwise op with broadcasting.
void register_binary(const std::string& name, DType dt,
                     float (*scalar)(float, float),
                     void (*vec)(const float*, const float*, float*, int64_t)) {
    Op op;
    op.name = name;
    op.dtype = dt;
    op.device = Device::cpu();
    op.forward = [name, scalar, vec](const TensorVec& in) -> TensorVec {
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
        Tensor neg = Tensor::empty(output_grads[0].shape(), DType::F32, inputs[0].device());
        cpu_neg(fdata(const_cast<Tensor&>(output_grads[0])), fdata(neg), output_grads[0].numel());
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
        run_binary_bwd_op(inputs[1], output_grads[0], g_a, [](float x, float y) { return x * y; });
        reduce_to_input_shape(g_a, inputs[0].shape(), input_grads[0]);
        Tensor g_b = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(inputs[0], output_grads[0], g_b, [](float x, float y) { return x * y; });
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
        Tensor g_a = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(output_grads[0], inputs[1], g_a, [](float x, float y) { return x / y; });
        reduce_to_input_shape(g_a, inputs[0].shape(), input_grads[0]);
        Tensor tmp1 = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(output_grads[0], inputs[0], tmp1, [](float x, float y) { return x * y; });
        Tensor tmp2 = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(inputs[1], inputs[1], tmp2, [](float x, float y) { return x * y; });
        Tensor g_b = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(tmp1, tmp2, g_b, [](float x, float y) { return x / y; });
        Tensor neg_gb = Tensor::empty(g_b.shape(), DType::F32, g_b.device());
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
        // d/da = b * a^(b-1) * og
        Tensor b_minus_one = Tensor::empty(inputs[1].shape(), DType::F32, inputs[1].device());
        Tensor ones = Tensor::ones(inputs[1].shape(), DType::F32, inputs[1].device());
        run_binary_bwd_op(inputs[1], ones, b_minus_one, [](float x, float y) { return x - y; });
        Tensor a_pow = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(inputs[0], b_minus_one, a_pow, [](float x, float y) { return std::pow(x, y); });
        Tensor tmp_ga = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(a_pow, inputs[1], tmp_ga, [](float x, float y) { return x * y; });
        Tensor g_a = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(tmp_ga, output_grads[0], g_a, [](float x, float y) { return x * y; });
        reduce_to_input_shape(g_a, inputs[0].shape(), input_grads[0]);

        // d/db = a^b * log(a) * og
        Tensor a_safe = Tensor::empty(inputs[0].shape(), DType::F32, inputs[0].device());
        int64_t na = inputs[0].numel();
        const float* ap = fdata(const_cast<Tensor&>(inputs[0]));
        float* a_safe_p = fdata(a_safe);
        for (int64_t i = 0; i < na; ++i) a_safe_p[i] = std::max(ap[i], 1e-12f);
        Tensor ab = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(a_safe, inputs[1], ab, [](float x, float y) { return std::pow(x, y); });
        Tensor log_a = Tensor::empty(inputs[0].shape(), DType::F32, inputs[0].device());
        cpu_log(fdata(a_safe), fdata(log_a), na);
        Tensor ab_log_a = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(ab, log_a, ab_log_a, [](float x, float y) { return x * y; });
        Tensor g_b = Tensor::empty(output_grads[0].shape(), DType::F32, output_grads[0].device());
        run_binary_bwd_op(ab_log_a, output_grads[0], g_b, [](float x, float y) { return x * y; });
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

void register_softmax() {
    Op op; op.name = "softmax"; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [](const TensorVec& in) -> TensorVec {
        const auto& a = in[0];
        int rank = static_cast<int>(a.shape().rank());
        if (rank < 1) throw std::runtime_error("softmax: need at least 1D");
        // Softmax along the last axis.
        int64_t C = a.shape().dims[rank - 1];  // number of classes
        int64_t rows = a.numel() / C;
        const float* ap = fdata(const_cast<Tensor&>(a));
        Tensor y = Tensor::empty(a.shape(), a.dtype(), a.device());
        float* yp = fdata(y);
        for (int64_t r = 0; r < rows; ++r) {
            const float* row_in = ap + r * C;
            float* row_out = yp + r * C;
            // Numerically-stable softmax: subtract max first.
            float mx = row_in[0];
            for (int64_t c = 1; c < C; ++c) if (row_in[c] > mx) mx = row_in[c];
            double s = 0.0;
            for (int64_t c = 0; c < C; ++c) {
                row_out[c] = std::exp(row_in[c] - mx);
                s += row_out[c];
            }
            for (int64_t c = 0; c < C; ++c) row_out[c] /= static_cast<float>(s);
        }
        return {y};
    };
    op.backward = [](const TensorVec& outputs, const TensorVec& output_grads,
                     const TensorVec& /*inputs*/, TensorVec& input_grads) {
        // softmax backward:  dx_i = y_i * (dy_i - sum_j(dy_j * y_j))
        const Tensor& y = outputs[0];
        const Tensor& og = output_grads[0];
        int rank = static_cast<int>(y.shape().rank());
        int64_t C = y.shape().dims[rank - 1];
        int64_t rows = y.numel() / C;
        const float* yp = fdata(const_cast<Tensor&>(y));
        const float* ogp = fdata(const_cast<Tensor&>(og));
        float* gp = fdata(input_grads[0]);
        for (int64_t r = 0; r < rows; ++r) {
            const float* yr = yp + r * C;
            const float* ogr = ogp + r * C;
            float* gr = gp + r * C;
            // dot = sum_j (og_j * y_j)
            double dot = 0.0;
            for (int64_t c = 0; c < C; ++c) dot += ogr[c] * yr[c];
            for (int64_t c = 0; c < C; ++c) {
                gr[c] += yr[c] * (ogr[c] - static_cast<float>(dot));
            }
        }
    };
    OpRegistry::instance().register_op("softmax", DType::F32, std::move(op));
}

void register_conv2d() {
    Op op; op.name = "conv2d"; op.dtype = DType::F32; op.device = Device::cpu();
    op.forward = [](const TensorVec& in) -> TensorVec {
        const auto& x = in[0];
        const auto& w = in[1];
        const auto& bias = in[2];
        const auto& stride_t = in[3];
        const auto& padding_t = in[4];
        
        if (x.shape().rank() != 4) throw std::runtime_error("conv2d: x must be 4D (N, C, H, W)");
        if (w.shape().rank() != 4) throw std::runtime_error("conv2d: weight must be 4D (C_out, C_in, kH, kW)");
        
        int64_t N = x.shape().dims[0];
        int64_t C_in = x.shape().dims[1];
        int64_t H = x.shape().dims[2];
        int64_t W = x.shape().dims[3];
        
        int64_t C_out = w.shape().dims[0];
        int64_t C_in_w = w.shape().dims[1];
        int64_t kH = w.shape().dims[2];
        int64_t kW = w.shape().dims[3];
        
        if (C_in != C_in_w) throw std::runtime_error("conv2d: weight channel mismatch");
        
        int stride = static_cast<int>(*fdata(const_cast<Tensor&>(stride_t)));
        int padding = static_cast<int>(*fdata(const_cast<Tensor&>(padding_t)));
        
        int64_t H_out = (H - kH + 2 * padding) / stride + 1;
        int64_t W_out = (W - kW + 2 * padding) / stride + 1;
        
        if (H_out <= 0 || W_out <= 0) throw std::runtime_error("conv2d: invalid output dimensions");
        
        Tensor out = Tensor::empty(Shape({N, C_out, H_out, W_out}), x.dtype(), x.device());
        
        const float* xp = fdata(const_cast<Tensor&>(x));
        const float* wp = fdata(const_cast<Tensor&>(w));
        const float* bp = fdata(const_cast<Tensor&>(bias));
        float* op_data = fdata(out);
        
        std::memset(op_data, 0, out.numel() * sizeof(float));
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t co = 0; co < C_out; ++co) {
                float b_val = bp[co];
                for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                    for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                        float sum = b_val;
                        for (int64_t ci = 0; ci < C_in; ++ci) {
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                int64_t h_in = h_out * stride - padding + kh;
                                if (h_in < 0 || h_in >= H) continue;
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    int64_t w_in = w_out * stride - padding + kw;
                                    if (w_in < 0 || w_in >= W) continue;
                                    
                                    int64_t x_idx = ((n * C_in + ci) * H + h_in) * W + w_in;
                                    int64_t w_idx = ((co * C_in + ci) * kH + kh) * kW + kw;
                                    sum += xp[x_idx] * wp[w_idx];
                                }
                            }
                        }
                        int64_t out_idx = ((n * C_out + co) * H_out + h_out) * W_out + w_out;
                        op_data[out_idx] = sum;
                    }
                }
            }
        }
        return {out};
    };
    op.backward = [](const TensorVec& outputs, const TensorVec& output_grads,
                     const TensorVec& inputs, TensorVec& input_grads) {
        const auto& x = inputs[0];
        const auto& w = inputs[1];
        const auto& stride_t = inputs[3];
        const auto& padding_t = inputs[4];
        
        const auto& og = output_grads[0];
        
        int64_t N = x.shape().dims[0];
        int64_t C_in = x.shape().dims[1];
        int64_t H = x.shape().dims[2];
        int64_t W = x.shape().dims[3];
        
        int64_t C_out = w.shape().dims[0];
        int64_t kH = w.shape().dims[2];
        int64_t kW = w.shape().dims[3];
        
        int stride = static_cast<int>(*fdata(const_cast<Tensor&>(stride_t)));
        int padding = static_cast<int>(*fdata(const_cast<Tensor&>(padding_t)));
        
        int64_t H_out = og.shape().dims[2];
        int64_t W_out = og.shape().dims[3];
        
        const float* xp = fdata(const_cast<Tensor&>(x));
        const float* wp = fdata(const_cast<Tensor&>(w));
        const float* ogp = fdata(const_cast<Tensor&>(og));
        
        float* dxp = fdata(input_grads[0]);
        float* dwp = fdata(input_grads[1]);
        float* dbiasp = fdata(input_grads[2]);
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t co = 0; co < C_out; ++co) {
                for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                    for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                        int64_t og_idx = ((n * C_out + co) * H_out + h_out) * W_out + w_out;
                        float og_val = ogp[og_idx];
                        
                        dbiasp[co] += og_val;
                        
                        for (int64_t ci = 0; ci < C_in; ++ci) {
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                int64_t h_in = h_out * stride - padding + kh;
                                if (h_in < 0 || h_in >= H) continue;
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    int64_t w_in = w_out * stride - padding + kw;
                                    if (w_in < 0 || w_in >= W) continue;
                                    
                                    int64_t x_idx = ((n * C_in + ci) * H + h_in) * W + w_in;
                                    int64_t w_idx = ((co * C_in + ci) * kH + kh) * kW + kw;
                                    
                                    dwp[w_idx] += og_val * xp[x_idx];
                                    dxp[x_idx] += og_val * wp[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    };
    OpRegistry::instance().register_op("conv2d", DType::F32, std::move(op));
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
        Tensor at = Tensor::empty(Shape({K, M}), a.dtype(), a.device());
        float* atp = fdata(at);
        const float* ap = fdata(a);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) atp[k * M + i] = ap[i * K + k];
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
    register_softmax();
    register_conv2d();
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
        fdata(dst)[dst_idx] += fdata(og)[idx];
    }
}

}  // namespace micrograd
