// Loss functions.
#pragma once
#include "micrograd/tensor.hpp"

namespace micrograd {

Tensor mse_loss(const Tensor& pred, const Tensor& target);
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target);
Tensor bce_loss(const Tensor& pred, const Tensor& target);

}  // namespace micrograd
