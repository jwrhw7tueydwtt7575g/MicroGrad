// Device: type + index. v0.1 only ships CPU.
#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>

namespace micrograd {

enum class DeviceType : uint8_t {
    CPU = 0,
    CUDA = 1,
};

struct Device {
    DeviceType type = DeviceType::CPU;
    int index = 0;

    bool operator==(const Device& o) const { return type == o.type && index == o.index; }
    bool operator!=(const Device& o) const { return !(*this == o); }

    static Device cpu() { return {DeviceType::CPU, 0}; }
    static Device cuda(int i = 0) { return {DeviceType::CUDA, i}; }

    std::string to_string() const {
        if (type == DeviceType::CPU) return "cpu";
        return "cuda:" + std::to_string(index);
    }
};

}  // namespace micrograd
