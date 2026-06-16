#include "micrograd/storage.hpp"
#include "micrograd/stream.hpp"
#include <stdexcept>
#include <cstring>

namespace micrograd {

void CPUStorage::copy_from(const Storage& src, const Stream& /*s*/) {
    if (src.nbytes() > bytes_) {
        throw std::runtime_error("CPUStorage::copy_from: src larger than dst");
    }
    std::memcpy(data_.data(), src.raw(), src.nbytes());
    if (src.nbytes() < bytes_) {
        std::memset(data_.data() + src.nbytes(), 0, bytes_ - src.nbytes());
    }
}

StorageHandle make_storage(Device d, size_t bytes) {
    if (d.type == DeviceType::CPU) {
        return std::make_shared<CPUStorage>(bytes);
    }
    throw std::runtime_error("CUDA storage not implemented in v0.1");
}

}  // namespace micrograd
