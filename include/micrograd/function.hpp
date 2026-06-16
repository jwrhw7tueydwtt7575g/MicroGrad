// Function: a Python-facing callable that holds a Graph. All four
// functional transforms (grad/vmap/jit/pmap) consume and produce Function
// objects.
#pragma once
#include "micrograd/ir.hpp"
#include "micrograd/tensor.hpp"
#include <vector>
#include <string>
#include <memory>

namespace micrograd {

class Function;
using FunctionPtr = std::shared_ptr<Function>;

class Function {
public:
    Function();
    explicit Function(Graph g);

    // Run the captured forward again. Used by grad/vmap/jit/pmap.
    // `bindings` is indexed by Graph::input_ids.
    TensorVec apply(const TensorVec& bindings);

    // Backward: single linear pass over reverse_order() with topological
    // release. For each leaf parameter, copies the accumulated grad into
    // param.grad().
    void backward(const Tensor& loss);

    // Functional transforms.
    FunctionPtr grad(std::vector<int> argnums);
    FunctionPtr vmap(int in_batch_dim = 0, int out_batch_dim = 0);
    FunctionPtr jit();
    FunctionPtr pmap();                 // v0.2

    // Serialization
    void save(const std::string& path) const;
    static FunctionPtr load(const std::string& path);

    // Accessor
    Graph& graph() { return graph_; }
    const Graph& graph() const { return graph_; }

    // Mark the active function for tracing. Returns a RAII guard that
    // restores the previous tracer.
    class Scope;
    Scope enter();

private:
    Graph graph_;
    std::vector<Tensor> input_slots_;
    std::vector<Tensor> output_slots_;
};

class Function::Scope {
public:
    Scope(Function* f);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    bool prev_recording;
    Function* prev_function;
    Graph* prev_graph;
};

}  // namespace micrograd
