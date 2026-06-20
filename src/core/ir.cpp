#include "micrograd/ir.hpp"
#include <stdexcept>

namespace micrograd {

int Graph::add_node() {
    int id = static_cast<int>(nodes.size());
    IRNode n;
    n.id = id;
    n.graph = this;
    nodes.push_back(std::move(n));
    return id;
}

std::vector<int> Graph::reverse_order() const {
    int n = static_cast<int>(nodes.size());
    std::vector<int> order;
    order.reserve(n);
    std::vector<uint8_t> visited(n, 0);

    // Post-order DFS on the input graph: recurse on inputs first, then
    // record the node. The resulting order is a topological sort in
    // forward direction. Backward then walks this list in reverse so the
    // output node is processed first and its successors' cached state has
    // not yet been released.
    std::function<void(int)> dfs = [&](int u) {
        if (u < 0 || u >= n) return;
        if (visited[u]) return;
        visited[u] = 1;
        if (u < static_cast<int>(nodes.size())) {
            for (int p : nodes[u].inputs) dfs(p);
        }
        order.push_back(u);
    };

    if (output_id >= 0) dfs(output_id);
    for (int i = 0; i < n; ++i) {
        if (!visited[i]) dfs(i);
    }
    // Reverse so backward starts at the output.
    std::reverse(order.begin(), order.end());
    return order;
}

std::vector<int> Graph::compute_refcounts() const {
    int n = static_cast<int>(nodes.size());
    std::vector<int> rc(n, 0);
    for (int u = 0; u < n; ++u) {
        for (int p : nodes[u].inputs) {
            if (p >= 0 && p < n) rc[p]++;
        }
    }
    // Each node's own backward_kernel needs its cached_inputs/outputs, so
    // add 1 to keep the cache alive until this node's backward runs.
    for (int u = 0; u < n; ++u) {
        if (nodes[u].backward_kernel) rc[u]++;
    }
    return rc;
}

Tracer& Tracer::current() {
    thread_local Tracer t;
    return t;
}

}  // namespace micrograd
