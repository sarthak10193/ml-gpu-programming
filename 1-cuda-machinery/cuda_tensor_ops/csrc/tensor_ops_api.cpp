#include <torch/extension.h>
#include "tensor_ops/linear_kernel.h"

namespace cuda_tensor_ops {
// Forward declaration of the function defined in linear.cpp
torch::Tensor linear_forward_torch(const torch::Tensor& input, const float scale, const float bias);
}

// Expose functions to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &cuda_tensor_ops::linear_forward_torch, "Linear Op")
}