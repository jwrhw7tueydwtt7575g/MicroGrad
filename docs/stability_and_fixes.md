# Engine Stability & Quick Fixes

During the development and rigorous end-to-end regression testing of MicroGrad, we identified and resolved several critical system bugs in the C++ core and optimization pipeline. This document records the debug history and fixes implemented to achieve 100% test-suite stability.

---

## 1. Eager Graph Isolation State Leak

### Symptom
Subsequent eager unit tests randomly crashed or failed with shape mismatches and memory access violations.

### Root Cause
MicroGrad uses a global `Tracer` class to record the active calculation graph during eager tensor operations. When calling `loss.backward()` in eager mode outside of an explicit `with Function():` context, the transient eager graph state was not cleaned up after the backward pass completed. This leaked active IR nodes and producers into the next test, contaminating its computation graph.

### Fix
Integrated eager graph lifetime management with active functions:
- Inside `py_backward` (`src/python/py_tensor.cpp`), we inspect if there is any active function context running (`active_function()`).
- If no active function context is running, the transient graph is considered complete. We explicitly reset the current `Tracer`'s graph pointer and function pointer, isolating subsequent test graphs.

---

## 2. Missing Softmax Operator Registry

### Symptom
When training an MLP on XOR or cross-entropy regression, loss stalled, gradients became zero, and parameter updates ceased.

### Root Cause
The `cross_entropy_loss` function in `src/core/loss.cpp` was directly calling a C++ Softmax utility class to compute forward logits. Because the Softmax call bypassed the central `run_op()` operator registry dispatch, it did not create an IR node. The autograd graph could not trace back through the softmax activation, completely blocking gradient flow back to the logits and MLP layer weights.

### Fix
Registered `softmax` as a first-class CPU operator:
- Added a proper Jacobian-based CPU backward kernel in `src/core/op_kernels.cpp`:
  $$\frac{\partial L}{\partial x_i} = y_i \left( \frac{\partial L}{\partial y_i} - \sum_{j} \frac{\partial L}{\partial y_j} y_j \right)$$
- Registered `softmax` in the global dispatch table.
- Refactored `cross_entropy_loss` to call `run_op("softmax")`, enabling correct gradient propagation.

---

## 3. Module Parameter Correlation (Layer Seed Collision)

### Symptom
Multilayer Perceptron (MLP) layers produced identical weights upon initialization, preventing the model from breaking symmetry and learning complex functions.

### Root Cause
The constructor for the `Linear` module in `src/core/linear.cpp` initialized its weights using a pseudo-random number generator (PRNG) with a hardcoded static seed. As a result, every `Linear` instance created within the same thread used the exact same random seed sequence, resulting in identical weight parameter values across all layers.

### Fix
Introduced dynamic seed offsets:
- Modified `src/core/linear.cpp` to track a `thread_local` counter of instantiated linear layers.
- Offset the base random seed by this counter to guarantee unique initialization seeds for every layer instance.

---

## 4. Optimization & Learning Rate Tuning

### Symptom
The `test_toy_regression` e2e test failed to converge to a loss $< 0.05$ within the default epoch limit.

### Root Cause
The initial learning rate was set to `1e-3` in the Adam optimizer. While safe for simple functions, it was too conservative to allow the MLP to escape a local loss plateau when fitting the non-linear $y = \sin(3x)$ curve, leading to training time-out failures.

### Fix
Adjusted optimizer parameters:
- Increased the learning rate in `tests/e2e/test_toy_regression.py` to `3e-2`.
- Increased training steps to `500`.
- This allowed the MLP to break out of flat gradients and consistently reach a loss $< 0.05$ across all runs.
