#include "micrograd/op_registry.hpp"
#include "micrograd/ir.hpp"
#include <stdexcept>
#include <sstream>

namespace micrograd {

OpRegistry& OpRegistry::instance() {
    static OpRegistry r;
    return r;
}

void OpRegistry::register_op(const std::string& name, DType dtype, Op op) {
    register_op(name, dtype, op.device, op);
}

void OpRegistry::register_op(const std::string& name, DType dtype, Device device, Op op) {
    std::lock_guard<std::mutex> lk(mu_);
    OpKey key{name, dtype, device};
    if (op.dtype != dtype) {
        throw std::runtime_error("OpRegistry::register_op: dtype mismatch for " + name);
    }
    if (op.device != device) {
        throw std::runtime_error("OpRegistry::register_op: device mismatch for " + name);
    }
    auto it = table_.find(key);
    if (it != table_.end()) {
        // Replace (used by user-op registration).
        *it->second = std::move(op);
    } else {
        table_.emplace(key, std::make_unique<Op>(std::move(op)));
    }
}

const Op* OpRegistry::find(const std::string& name, DType dtype, Device device) const {
    std::lock_guard<std::mutex> lk(mu_);
    OpKey key{name, dtype, device};
    auto it = table_.find(key);
    if (it == table_.end()) return nullptr;
    return it->second.get();
}

bool OpRegistry::has(const std::string& name, Device device) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& kv : table_) {
        if (kv.first.name == name && kv.first.device == device) return true;
    }
    return false;
}

TensorVec run_op(const std::string& name, const TensorVec& inputs,
                 const std::vector<uint8_t>& attr_bytes) {
    if (inputs.empty()) {
        throw std::runtime_error("run_op: empty inputs");
    }
    DType dtype = inputs[0].dtype();
    Device device = inputs[0].device();
    for (const auto& t : inputs) {
        if (t.dtype() != dtype) {
            throw std::runtime_error("run_op: dtype mismatch on " + name);
        }
        if (t.device() != device) {
            throw std::runtime_error("run_op: device mismatch on " + name);
        }
    }
    const Op* op = OpRegistry::instance().find(name, dtype, device);
    if (!op) {
        std::ostringstream s;
        s << "run_op: no op registered for name=" << name
          << " dtype=" << dtype_name(dtype) << " device=" << device.to_string();
        throw std::runtime_error(s.str());
    }
    TensorVec outputs = op->forward(inputs);

    // Record into active graph if tracing.
    Tracer& tr = Tracer::current();
    if (tr.recording && tr.graph) {
        // For each input that has requires_grad=true and is not yet a
        // parameter in the active graph, register it as a param node.
        for (size_t k = 0; k < inputs.size(); ++k) {
            const Tensor& in = inputs[k];
            if (in.requires_grad() && in.producer() == nullptr) {
                int pid = tr.graph->add_node();
                IRNode& pnode = tr.graph->node(pid);
                pnode.op_name = "param";
                pnode.dtype = in.dtype();
                pnode.device = in.device();
                pnode.cached_inputs = {in};
                pnode.cached_outputs = {in};
                tr.graph->param_ids.push_back(pid);
                // Re-tag the input tensor's producer so this op references it.
                const_cast<Tensor&>(in).set_producer(&pnode);
            }
        }

        int id = tr.graph->add_node();
        IRNode& node = tr.graph->node(id);
        node.op_name = name;
        node.dtype = dtype;
        node.device = device;
        for (const auto& in : inputs) {
            int p = in.producer() ? in.producer()->id : -1;
            node.inputs.push_back(p);
        }
        node.cached_inputs = inputs;
        node.attrs = attr_bytes.empty() ? op->attr_bytes : attr_bytes;
        node.backward_kernel = op->backward;
        node.cached_outputs = outputs;
        for (auto& out : outputs) {
            out.set_producer(&tr.graph->node(id));
        }
        tr.current_node = id;
    }

    return outputs;
}

}  // namespace micrograd
