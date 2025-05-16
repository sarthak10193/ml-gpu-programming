#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>


enum class DataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2
};

namespace cuda_tensor_ops {

// Forward declaration of the CUDA kernel launcher
// Template parameters allow for compile-time optimizations
template <typename T, int BLOCK_SIZE>
void launch_linear_kernel(
    T* output,      // Output tensor
    const T* input, // Input tensor
    const float scale,  // Multiplicative factor (a)
    const float bias,   // Additive bias (b)
    const int batch_size,
    const int seq_length,
    cudaStream_t stream
);

// Host function to choose template parameters at runtime
void linear_forward(
    void* output,
    const void* input,
    const float scale,
    const float bias,
    const int batch_size,
    const int seq_length,
    const DataType data_type,  // 0: fp32, 1: fp16, 2: bf16
    cudaStream_t stream
);

} // namespace cuda_tensor_ops