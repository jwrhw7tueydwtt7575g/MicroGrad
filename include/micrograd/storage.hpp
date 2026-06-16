// Storage: ref-counted per-device memory. v0.1 only has CPUStorage.
#pragma once
#include "micrograd/device.hpp"
#include "micrograd/stream.hpp"
#include "micrograd/dtype.hpp"
#include <cstdint>
#include <memory>
#include <vector>
#include <cstring>

namespace micrograd {

class Storage {
public:
    virtual ~Storage() = default;
    virtual Device device() const = 0;
    virtual size_t nbytes() const = 0;
    virtual void* raw() = 0;
    virtual const void* raw() const = 0;
    virtual void copy_from(const Storage& src, const Stream& s) = 0;
};

class CPUStorage : public Storage {
public:
    CPUStorage(size_t bytes) : bytes_(bytes), data_(bytes, 0) {}
    Device device() const override { return Device::cpu(); }
    size_t nbytes() const override { return bytes_; }
    void* raw() override { return data_.data(); }
    const void* raw() const override { return data_.data(); }
    void copy_from(const Storage& src, const Stream& /*s*/) override;
private:
    size_t bytes_;
    std::vector<uint8_t> data_;
};

using StorageHandle = std::shared_ptr<Storage>;
StorageHandle make_storage(Device d, size_t bytes);

}  // namespace micrograd
