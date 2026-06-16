#include "micrograd/autograd.hpp"
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace micrograd {

void backward(Function& f, const Tensor& loss) {
    Graph& g = f.graph();
    int n = static_cast<int>(g.nodes.size());
    if (n == 0) return;

    // Determine the output node: prefer g.output_id if set, else use the
    // loss tensor's producer.
    int output_id = g.output_id;
    if (output_id < 0 && loss.producer()) {
        output_id = loss.producer()->id;
    }
    if (output_id < 0) {
        // No graph; nothing to do.
        return;
    }
    g.output_id = output_id;

    std::vector<int> order = g.reverse_order();

    std::unordered_map<int, Tensor> grad_for_node;
    grad_for_node.reserve(n);

    Tensor seed = Tensor::ones(loss.shape(), loss.dtype(), loss.device());
    grad_for_node[output_id] = seed;

    std::vector<int> refcounts = g.compute_refcounts();
    std::vector<bool> is_param(n, false);
    for (int pid : g.param_ids) if (pid >= 0 && pid < n) is_param[pid] = true;

    for (int id : order) {
        auto it = grad_for_node.find(id);
        if (it == grad_for_node.end()) continue;
        IRNode& node = g.node(id);
        if (!node.backward_kernel) continue;

        TensorVec in_grads(node.cached_inputs.size());
        for (size_t i = 0; i < in_grads.size(); ++i) {
            in_grads[i] = Tensor::zeros(
                node.cached_inputs[i].shape(),
                node.cached_inputs[i].dtype(),
                node.cached_inputs[i].device());
        }

        node.backward_kernel(node.cached_outputs, {it->second}, node.cached_inputs, in_grads);

        for (size_t i = 0; i < node.inputs.size(); ++i) {
            int p = node.inputs[i];
            if (p < 0) continue;
            auto git = grad_for_node.find(p);
            if (git == grad_for_node.end()) {
                grad_for_node.emplace(p, std::move(in_grads[i]));
            } else {
                auto& dst = git->second;
                auto& src = in_grads[i];
                if (dst.dtype() != src.dtype() || dst.shape() != src.shape()) {
                    throw std::runtime_error("backward: grad shape/dtype mismatch");
                }
                int64_t N = dst.numel();
                float* dp = static_cast<float*>(dst.data_ptr());
                const float* sp = static_cast<const float*>(src.data_ptr());
                for (int64_t k = 0; k < N; ++k) dp[k] += sp[k];
            }
            if (!is_param[p]) {
                refcounts[p]--;
                if (refcounts[p] == 0) {
                    g.node(p).cached_inputs.clear();
                    g.node(p).cached_outputs.clear();
                }
            }
        }
        node.cached_outputs.clear();
    }

    // Copy grads into leaf parameters.
    for (int pid : g.param_ids) {
        if (pid < 0 || pid >= n) continue;
        auto it = grad_for_node.find(pid);
        if (it == grad_for_node.end()) continue;
        IRNode& pnode = g.node(pid);
        if (pnode.cached_inputs.empty()) continue;
        Tensor& p = pnode.cached_inputs[0];
        p.init_grad();
        int64_t N = p.numel();
        const float* sp = static_cast<const float*>(it->second.data_ptr());
        float* dp = static_cast<float*>(p.grad().data_ptr());
        for (int64_t k = 0; k < N; ++k) dp[k] = sp[k];
    }
}

}  // namespace micrograd
