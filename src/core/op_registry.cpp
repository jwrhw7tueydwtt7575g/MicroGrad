#include "micrograd/op_registry.hpp"
#include "micrograd/ir.hpp"
#include "micrograd/function.hpp"
#include <cstdio>
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
    for (const auto& kv : table_) {
        if (kv.first.name == name && kv.first.device == device) return true;
    }
    return false;
}

// Internal core implementation: takes pointers to avoid copying Tensors so
// updates to grad_/producer_ on the original objects are visible. This is
// the path used by the Python bindings; pure-C++ callers can use the value-
// overload below which simply builds a pointer vector.
TensorVec run_op_impl(const std::string& name,
                      const std::vector<const Tensor*>& inputs,
                      const std::vector<uint8_t>& attr_bytes) {
    if (inputs.empty()) {
        throw std::runtime_error("run_op: empty inputs");
    }
    DType dtype = inputs[0]->dtype();
    Device device = inputs[0]->device();
    for (const auto* t : inputs) {
        if (t->dtype() != dtype) {
            throw std::runtime_error("run_op: dtype mismatch on " + name);
        }
        if (t->device() != device) {
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
    // op->forward expects a TensorVec; build it from the pointers. This
    // copies the Tensors into the vector, which is fine for forward but the
    // copies share grad_/data_ via shared_ptr.
    TensorVec input_vec;
    input_vec.reserve(inputs.size());
    for (const Tensor* t : inputs) input_vec.push_back(*t);
    TensorVec outputs = op->forward(input_vec);

    // Propagate requires_grad: if any input requires grad, the outputs
    // should too (standard autograd behavior).
    bool any_in_grad = false;
    for (const Tensor* in : inputs) {
        if (in->requires_grad()) { any_in_grad = true; break; }
    }
    if (any_in_grad) {
        for (auto& out : outputs) {
            if (!out.requires_grad()) out.set_requires_grad(true);
        }
    }

    // Record into active graph if tracing.
    Tracer& tr = Tracer::current();
    if (tr.recording) {
        if (!tr.graph) {
            auto fn_ptr = std::make_shared<Function>();
            tr.graph = &fn_ptr->graph();
            tr.graph->owner = fn_ptr.get();
            tr.graph->owner_keepalive = fn_ptr;
            tr.function = fn_ptr.get();
        }
        if (tr.graph) {
            // For each input that has requires_grad=true and is not yet a
            // parameter in the active graph, register it as a param node.
            // init_grad() is called on the ORIGINAL Tensor (via the pointer)
            // so the Python-side holder sees the grad allocation.
            for (size_t k = 0; k < inputs.size(); ++k) {
                const Tensor* in_ptr = inputs[k];
                Tensor& in = const_cast<Tensor&>(*in_ptr);
                if (in.requires_grad() && (!in.has_producer() || in.producer().graph != tr.graph)) {
                    in.init_grad();

                    int pid = tr.graph->add_node();
                    IRNode& pnode = tr.graph->node(pid);
                    pnode.op_name = "param";
                    pnode.dtype = in.dtype();
                    pnode.device = in.device();
                    pnode.cached_inputs = {in};
                    pnode.cached_outputs = {in};
                    tr.graph->param_ids.push_back(pid);
                    in.set_producer(tr.graph, pid);
                }
            }

            int id = tr.graph->add_node();
            IRNode& node = tr.graph->node(id);
            node.op_name = name;
            node.dtype = dtype;
            node.device = device;
            for (const Tensor* in : inputs) {
                int p = (in->has_producer() && in->producer().graph == tr.graph) ? in->producer().node_id : -1;
                node.inputs.push_back(p);
            }
            // Copy inputs for cached storage (shared_ptr semantics keep grad_
            // and data_ shared with the originals).
            TensorVec cached;
            cached.reserve(inputs.size());
            for (const Tensor* t : inputs) cached.push_back(*t);
            node.cached_inputs = std::move(cached);
            node.attrs = attr_bytes.empty() ? op->attr_bytes : attr_bytes;
            node.backward_kernel = op->backward;
            node.cached_outputs = outputs;
            for (auto& out : outputs) {
                out.set_producer(tr.graph, id);
            }
            tr.current_node = id;
        }
    }

    return outputs;
}

TensorVec run_op_ptrs(const std::string& name,
                      const std::vector<const Tensor*>& inputs,
                      const std::vector<uint8_t>& attr_bytes) {
    return run_op_impl(name, inputs, attr_bytes);
}

TensorVec run_op(const std::string& name, const TensorVec& inputs,
                 const std::vector<uint8_t>& attr_bytes) {
    std::vector<const Tensor*> ptrs;
    ptrs.reserve(inputs.size());
    for (const auto& t : inputs) ptrs.push_back(&t);
    return run_op_impl(name, ptrs, attr_bytes);
}

}  // namespace micrograd