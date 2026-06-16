// Supported dtypes. v0.1 is float32-only for compute; int32 for labels.
#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>

namespace micrograd {

enum class DType : uint8_t {
    F32 = 0,
    F16 = 1,
    I32 = 2,
    I64 = 3,
};

inline const char* dtype_name(DType d) {
    switch (d) {
        case DType::F32: return "float32";
        case DType::F16: return "float16";
        case DType::I32: return "int32";
        case DType::I64: return "int64";
    }
    return "unknown";
}

inline size_t dtype_size(DType d) {
    switch (d) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::I32: return 4;
        case DType::I64: return 8;
    }
    return 0;
}

}  // namespace micrograd
