#include "linear_kernel.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace cuda_tensor_ops {

// ****************** 1. Kernel Header File ****************** //
// CUDA kernel that performs a fused linear transformation (ax + b)
template <typename T>
__global__ void linear_kernel(
    T* output,
    const T* input,
    const float scale,
    const float bias,
    const int batch_size,
    const int seq_length
) {
    // Calculate global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // out of bounds check
    if (idx < batch_size * seq_length) {
        // Apply linear transformation: ax + b
        output[idx] = static_cast<T>(static_cast<float>(input[idx]) * scale + bias);
    }
}

// ****************** 2. Kernel Launch ****************** //
// Kernel launcher with template specialization
// This pattern is common for different data types and block sizes
template <typename T, int BLOCK_SIZE>
void launch_linear_kernel(
    T* output,
    const T* input,
    const float scale,
    const float bias,
    const int batch_size,
    const int seq_length,
    cudaStream_t stream
) {
    // Calculate grid size (number of blocks)
    // Ensure we have enough threads to process all elements
    const int grid_size = (batch_size * seq_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch the kernel
    linear_kernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        output, input, scale, bias, batch_size, seq_length
    );
}

// ****************** 3. Explicit Template Instantiation ************** // 
// Explicit template instantiations for different types and block sizes
// This generates optimized code at compile time
#define INSTANTIATE_LAUNCH_KERNEL(T, BLOCK_SIZE) \
    template void launch_linear_kernel<T, BLOCK_SIZE>( \
        T* output, \
        const T* input, \
        const float scale, \
        const float bias, \
        const int batch_size, \
        const int seq_length, \
        cudaStream_t stream \
    );

// Instantiate for float (FP32)
INSTANTIATE_LAUNCH_KERNEL(float, 128)
INSTANTIATE_LAUNCH_KERNEL(float, 256)

// Instantiate for half (FP16)
INSTANTIATE_LAUNCH_KERNEL(__half, 128)
INSTANTIATE_LAUNCH_KERNEL(__half, 256)

// Instantiate for bfloat16 (BF16)
INSTANTIATE_LAUNCH_KERNEL(__nv_bfloat16, 128)
INSTANTIATE_LAUNCH_KERNEL(__nv_bfloat16, 256)

// ****************** 4. C++ Interface for Kernel Launching ************** //
// C++ interface that dispatches to the appropriate templated implementation
void linear_forward(
    void* output,
    const void* input,
    const float scale,
    const float bias,
    const int batch_size,
    const int seq_length,
    const int data_type,  // 0: fp32, 1: fp16, 2: bf16
    cudaStream_t stream
) {
    // Choose block size based on problem size
    // For simplicity, we're using a fixed heuristic here
    // In a real implementation, you might tune this based on more factors
    const int block_size = (batch_size * seq_length > 10000) ? 256 : 128;
    
    // Dispatch based on data type and block size
    if (data_type == 0) {  // FP32
        if (block_size == 256) {
            launch_linear_kernel<float, 256>(
                static_cast<float*>(output),
                static_cast<const float*>(input),
                scale, bias, batch_size, seq_length, stream
            );
        } else {
            launch_linear_kernel<float, 128>(
                static_cast<float*>(output),
                static_cast<const float*>(input),
                scale, bias, batch_size, seq_length, stream
            );
        }
    } else if (data_type == 1) {  // FP16
        if (block_size == 256) {
            launch_linear_kernel<__half, 256>(
                static_cast<__half*>(output),
                static_cast<const __half*>(input),
                scale, bias, batch_size, seq_length, stream
            );
        } else {
            launch_linear_kernel<__half, 128>(
                static_cast<__half*>(output),
                static_cast<const __half*>(input),
                scale, bias, batch_size, seq_length, stream
            );
        }
    } else if (data_type == 2) {  // BF16
        if (block_size == 256) {
            launch_linear_kernel<__nv_bfloat16, 256>(
                static_cast<__nv_bfloat16*>(output),
                static_cast<const __nv_bfloat16*>(input),
                scale, bias, batch_size, seq_length, stream
            );
        } else {
            launch_linear_kernel<__nv_bfloat16, 128>(
                static_cast<__nv_bfloat16*>(output),
                static_cast<const __nv_bfloat16*>(input),
                scale, bias, batch_size, seq_length, stream
            );
        }
    }
}

} // namespace cuda_tensor_ops