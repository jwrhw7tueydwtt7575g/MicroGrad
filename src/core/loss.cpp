#include "micrograd/loss.hpp"
#include "micrograd/op_registry.hpp"
#include <cmath>

namespace micrograd {

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    // mean((pred - target)^2)
    Tensor diff = run_op("sub", {pred, target})[0];
    Tensor sq = run_op("pow", {diff, Tensor::full(diff.shape(), 2.0f, diff.dtype(), diff.device())})[0];
    Tensor mean = run_op("mean", {sq})[0];
    return mean;
}

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target) {
    // Simplified: -log(softmax(logits)[target_class]). v0.1 expects target
    // shape == logits shape (one-hot).
    Tensor sm = Softmax(-1).forward(logits);
    Tensor eps = Tensor::full(sm.shape(), 1e-12f, sm.dtype(), sm.device());
    Tensor sm_safe = run_op("add", {sm, eps})[0];
    Tensor log_sm = run_op("log", {sm_safe})[0];
    Tensor neg = run_op("neg", {log_sm})[0];
    Tensor prod = run_op("mul", {neg, target})[0];
    return run_op("mean", {prod})[0];
}

Tensor bce_loss(const Tensor& pred, const Tensor& target) {
    // -[target*log(pred) + (1-target)*log(1-pred)]
    Tensor eps = Tensor::full(pred.shape(), 1e-12f, pred.dtype(), pred.device());
    Tensor p_safe = run_op("add", {pred, eps})[0];
    Tensor one = Tensor::ones(pred.shape(), pred.dtype(), pred.device());
    Tensor one_m_p = run_op("sub", {one, p_safe})[0];
    Tensor log_p = run_op("log", {p_safe})[0];
    Tensor log_1mp = run_op("log", {one_m_p})[0];
    Tensor tlp = run_op("mul", {target, log_p})[0];
    Tensor om_t = run_op("sub", {one, target})[0];
    Tensor term2 = run_op("mul", {om_t, log_1mp})[0];
    Tensor sum = run_op("add", {tlp, term2})[0];
    Tensor neg = run_op("neg", {sum})[0];
    return run_op("mean", {neg})[0];
}

}  // namespace micrograd
