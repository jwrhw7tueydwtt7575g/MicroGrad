#include "micrograd/active_fn.hpp"
#include "micrograd/ir.hpp"

namespace micrograd {

namespace {
thread_local Function* g_active = nullptr;
}

void set_active_function(Function* f) {
    g_active = f;
    Tracer& tr = Tracer::current();
    tr.function = f;
    tr.graph = f ? &f->graph() : nullptr;
    tr.recording = (f != nullptr);
}

Function* active_function() { return g_active; }

}  // namespace micrograd
