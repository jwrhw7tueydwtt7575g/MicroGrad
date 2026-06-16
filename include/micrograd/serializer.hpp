// Serialization: versioned JSON + content-addressed blob store.
// This is the v0.1 save/load path. v0.2 will move to a flatbuffer schema.
#pragma once
#include "micrograd/function.hpp"
#include "micrograd/optimizer.hpp"
#include <string>
#include <memory>
#include <unordered_map>

namespace micrograd {

class Serializer {
public:
    // Save the captured IR (no tensor data; only shapes/dtypes/ops).
    static void save_function(const Function& f, const std::string& path);
    // Load the captured IR. Tensor data is not restored.
    static FunctionPtr load_function(const std::string& path);

    // Save a tensor's raw bytes under a content-addressed (BLAKE3-like hash)
    // name. Returns the hash.
    static std::string save_blob(const Tensor& t, const std::string& dir);
    // Load a tensor by hash from a content-addressed store.
    static Tensor load_blob(const std::string& dir, const std::string& hash,
                            Shape shape, DType dtype, Device device = Device::cpu());

    // Optimizer state: serialize per-parameter moment tensors as blobs.
    static void save_optimizer(const Optimizer& opt, const std::string& path);
    static void load_optimizer(Optimizer& opt, const std::string& path);
};

}  // namespace micrograd
