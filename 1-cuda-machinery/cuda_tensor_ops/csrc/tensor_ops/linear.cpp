#include "linear_kernel.h"
#include <torch/extension.h>

namespace cuda_tensor_ops {

// C++ wrapper for the CUDA kernel that can be called from Python
// This handles error checking and data type conversion
torch::Tensor linear_forward_torch(
    const torch::Tensor& input,
    const float scale,
    const float bias
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    
    // Create output tensor with same properties as input
    auto output = torch::empty_like(input);
    
    // Get tensor dimensions
    const int batch_size = input.size(0);
    const int seq_length = input.size(1);
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Determine data type
    int data_type;
    if (input.scalar_type() == torch::ScalarType::Float) {
        data_type = 0;  // FP32
    } else if (input.scalar_type() == torch::ScalarType::Half) {
        data_type = 1;  // FP16
    } else if (input.scalar_type() == torch::ScalarType::BFloat16) {
        data_type = 2;  // BF16
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
    
    // Call the CUDA kernel
    linear_forward(
        output.data_ptr(),
        input.data_ptr(),
        scale,
        bias,
        batch_size,
        seq_length,
        data_type,
        stream
    );
    
    return output;
}

} // namespace cuda_tensor_ops