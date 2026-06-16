#include "micrograd/stream.hpp"
#include <memory>
#include <unordered_map>

namespace micrograd {

namespace {
CPUStream g_cpu_stream;
}

Stream& current_stream(Device d) {
    (void)d;
    return g_cpu_stream;
}

const Stream& current_stream_const(Device d) {
    (void)d;
    return g_cpu_stream;
}

}  // namespace micrograd
