#include "micrograd/op.hpp"
#include <functional>

namespace micrograd {

size_t OpKeyHash::operator()(const OpKey& k) const noexcept {
    size_t h = std::hash<std::string>{}(k.name);
    h ^= static_cast<size_t>(k.dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= (static_cast<size_t>(k.device.type) * 31u + static_cast<size_t>(k.device.index))
         + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

}  // namespace micrograd
