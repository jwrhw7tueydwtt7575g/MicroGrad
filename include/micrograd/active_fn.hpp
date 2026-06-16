// Globally register the active Function for autograd.
#pragma once
#include "micrograd/function.hpp"
#include <memory>

namespace micrograd {

// Set the thread-local active Function (used during a training step).
// Pass nullptr to deactivate. The pointer is borrowed; do not destroy it
// while a step is in progress.
void set_active_function(Function* f);
Function* active_function();

}  // namespace micrograd
