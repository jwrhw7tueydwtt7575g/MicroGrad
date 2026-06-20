// IR: the captured forward graph. Backward is a single linear pass over
// reverse_order() with topological release.
#pragma once
#include "micrograd/tensor.hpp"
#include "micrograd/dtype.hpp"
#include "micrograd/device.hpp"
#include "micrograd/op.hpp"
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace micrograd {

class Function;     // forward decl; resolved in micrograd/function.hpp
struct Graph;       // forward decl; defined below

struct IRNode {
    int id = -1;                                              // dense index in Graph::nodes
    std::string op_name;                                      // "add", "matmul", ...
    DType dtype = DType::F32;
    Device device = Device::cpu();
    std::vector<int> inputs;                                  // IRNode ids (or -1 for placeholders)
    std::vector<int> outputs;                                 // IRNode ids (filled in if outputs are named)
    std::vector<Tensor> cached_outputs;                       // forward outputs (kept for backward)
    std::vector<Tensor> cached_inputs;                        // forward inputs (kept for backward)
    std::vector<uint8_t> attrs;                               // op-specific

    // Back-pointer to the owning Graph. Set by Graph::add_node(). Allows
    // Tensor::producer() callers to navigate back to the Function that owns
    // the graph (via Graph::owner), which is needed by py_backward when no
    // active Function is set on the tracer.
    Graph* graph = nullptr;

    // Backward closure bound at capture time. Signature matches BackwardFn but
    // also receives attrs. Internally the Graph calls this with the grad
    // accumulator. We re-use BackwardFn for simplicity; the bound closure
    // captures `attrs` from the Op.
    BackwardFn backward_kernel;
};

struct Graph {
    std::vector<IRNode> nodes;
    std::vector<int> param_ids;                               // indices of leaf parameter nodes
    std::vector<int> input_ids;                               // placeholder ids (data inputs)
    int output_id = -1;                                       // IRNode id of the loss/output

    // Owning Function (set when this graph is wrapped by a Function). Used
    // by py_backward to dispatch backward() on the right Function when the
    // user calls Tensor.backward() outside a `with Function():` block.
    class Function* owner = nullptr;

    // Strong reference to keep a heap-allocated Function alive when this
    // graph was created ad-hoc by run_op (e.g. eager Python path that does
    // not enter an explicit Function context). Holds the owner alive.
    std::shared_ptr<class Function> owner_keepalive;

    // Allocate a fresh node id and a default-constructed IRNode.
    int add_node();
    IRNode& node(int id) { return nodes[id]; }
    const IRNode& node(int id) const { return nodes[id]; }

    // Reverse-topological order; computed once at backward() time.
    std::vector<int> reverse_order() const;

    // Successor count per node, computed for memory release.
    std::vector<int> compute_refcounts() const;
};

// Thread-local Tracer: when recording is on, run_op() pushes an IRNode onto
// the active Function (which owns a Graph). The Python wrapper pushes a
// fresh Function per training step.
struct Tracer {
    bool recording = true;
    class Function* function = nullptr;        // not owned; managed by Python
    Graph* graph = nullptr;                    // alias of function->graph()
    int current_node = -1;                     // last produced node id

    static Tracer& current();
};

}  // namespace micrograd
