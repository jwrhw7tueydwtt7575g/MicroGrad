// Stream: per-device execution context. v0.1 uses a single CPU stream.
#pragma once
#include "micrograd/device.hpp"
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include <functional>

namespace micrograd {

class Stream {
public:
    virtual ~Stream() = default;
    virtual Device device() const = 0;
    virtual void synchronize() const = 0;
};

class CPUStream : public Stream {
public:
    Device device() const override { return Device::cpu(); }
    void synchronize() const override {}
};

// Thread-local current stream. v0.1 always returns a shared CPU stream.
Stream& current_stream(Device d);
const Stream& current_stream_const(Device d);

}  // namespace micrograd
