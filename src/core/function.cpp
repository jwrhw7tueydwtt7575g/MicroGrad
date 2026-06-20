#include "micrograd/function.hpp"
#include "micrograd/serializer.hpp"
#include "micrograd/autograd.hpp"
#include "micrograd/op_registry.hpp"
#include "micrograd/op.hpp"
#include <utility>
#include <unordered_map>
#include <algorithm>
#include <string>

namespace micrograd {

Function::Function() = default;

Function::Function(const Graph& g) : graph_(g) {
    // Claim ownership of the graph so callers can navigate from a leaf
    // tensor's IRNode back to the Function that owns its graph.
    graph_.owner = this;
}

Function::Scope Function::enter() {
    return Scope(this);
}

Function::Scope::Scope(Function* f) {
    Tracer& tr = Tracer::current();
    prev_recording = tr.recording;
    prev_function = tr.function;
    prev_graph = tr.graph;
    tr.recording = true;
    tr.function = f;
    tr.graph = &f->graph();
}

Function::Scope::~Scope() {
    Tracer& tr = Tracer::current();
    tr.recording = prev_recording;
    tr.function = prev_function;
    tr.graph = prev_graph;
}

TensorVec Function::apply(const TensorVec& bindings) {
    input_slots_ = bindings;
    output_slots_.clear();

    Tracer& tr = Tracer::current();
    bool prev_rec = tr.recording;
    Function* prev_fn = tr.function;
    Graph* prev_g = tr.graph;
    tr.recording = false;
    tr.function = this;
    tr.graph = &graph_;

    std::unordered_map<int, Tensor> live;
    live.reserve(graph_.nodes.size());

    // Bind inputs and params.
    size_t bi = 0;
    for (int iid : graph_.input_ids) {
        if (bi >= bindings.size()) {
            tr.recording = prev_rec; tr.function = prev_fn; tr.graph = prev_g;
            throw std::runtime_error("Function::apply: not enough bindings (input)");
        }
        live[iid] = bindings[bi++];
    }
    for (int pid : graph_.param_ids) {
        if (bi >= bindings.size()) {
            tr.recording = prev_rec; tr.function = prev_fn; tr.graph = prev_g;
            throw std::runtime_error("Function::apply: not enough bindings (param)");
        }
        live[pid] = bindings[bi++];
    }

    for (auto& node : graph_.nodes) {
        if (live.count(node.id)) continue;
        if (std::find(graph_.input_ids.begin(), graph_.input_ids.end(), node.id) != graph_.input_ids.end()) continue;
        if (std::find(graph_.param_ids.begin(), graph_.param_ids.end(), node.id) != graph_.param_ids.end()) continue;

        const Op* op = OpRegistry::instance().find(node.op_name, node.dtype, node.device);
        if (!op) {
            tr.recording = prev_rec; tr.function = prev_fn; tr.graph = prev_g;
            throw std::runtime_error("Function::apply: op not registered: " + node.op_name);
        }
        TensorVec in_tensors;
        in_tensors.reserve(node.cached_inputs.size());
        for (size_t k = 0; k < node.cached_inputs.size(); ++k) {
            int p = node.inputs[k];
            auto it = live.find(p);
            if (it == live.end()) {
                tr.recording = prev_rec; tr.function = prev_fn; tr.graph = prev_g;
                throw std::runtime_error("Function::apply: missing live input");
            }
            in_tensors.push_back(it->second);
        }
        TensorVec outs = op->forward(in_tensors);
        if (!outs.empty()) live[node.id] = std::move(outs[0]);
    }

    TensorVec result;
    if (graph_.output_id >= 0) {
        auto it = live.find(graph_.output_id);
        if (it != live.end()) result.push_back(it->second);
    }
    output_slots_ = result;
    tr.recording = prev_rec;
    tr.function = prev_fn;
    tr.graph = prev_g;
    return result;
}

void Function::backward(const Tensor& loss) {
    micrograd::backward(*this, loss);
}

FunctionPtr Function::grad(std::vector<int> /*argnums*/) {
    return std::make_shared<Function>(graph_);
}

FunctionPtr Function::vmap(int /*in_batch_dim*/, int /*out_batch_dim*/) {
    return std::make_shared<Function>(graph_);
}

FunctionPtr Function::jit() {
    return std::make_shared<Function>(graph_);
}

FunctionPtr Function::pmap() {
    return std::make_shared<Function>(graph_);
}

void Function::save(const std::string& path) const {
    Serializer::save_function(*this, path);
}

FunctionPtr Function::load(const std::string& path) {
    return Serializer::load_function(path);
}

}  // namespace micrograd

