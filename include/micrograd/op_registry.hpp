// OpRegistry: dispatch table indexed by (name, dtype, device).
#pragma once
#include "micrograd/op.hpp"
#include <unordered_map>
#include <memory>
#include <mutex>

namespace micrograd {

class OpRegistry {
public:
    static OpRegistry& instance();

    // Register a CPU op for (name, dtype).
    void register_op(const std::string& name, DType dtype, Op op);
    void register_op(const std::string& name, DType dtype, Device device, Op op);

    // Find op; returns nullptr if missing.
    const Op* find(const std::string& name, DType dtype, Device device) const;

    // True if at least one op is registered for this name on this device.
    bool has(const std::string& name, Device device) const;

private:
    OpRegistry() = default;
    mutable std::mutex mu_;
    std::unordered_map<OpKey, std::unique_ptr<Op>, OpKeyHash> table_;
};

// Helper to run an op by name. Records an IR node if a Tracer is active.
// All inputs must share dtype and device; raises otherwise.
TensorVec run_op(const std::string& name, const TensorVec& inputs,
                 const std::vector<uint8_t>& attr_bytes = {});

}  // namespace micrograd
