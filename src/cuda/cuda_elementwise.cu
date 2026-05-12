/**
 * CUDA Element-wise Operations
 *
 * Implements element-wise tensor operations using custom CUDA kernels.
 * Includes vectorized versions for better memory bandwidth utilization.
 */

#ifdef DNN_ENABLE_CUDA

#include "dnn/cuda/cuda_common.hpp"
#include "dnn/cuda/cuda_ops.hpp"
#include "dnn/cuda/cuda_tensor.hpp"
#include <cuda_runtime.h>

namespace dnn {
namespace cuda {

// ============================================================================
// Basic Element-wise Kernels
// ============================================================================

/**
 * Element-wise addition kernel: c = a + b
 */
template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * Element-wise multiplication kernel: c = a * b
 */
template<typename T>
__global__ void mul_kernel(const T* a, const T* b, T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

/**
 * Scalar multiplication kernel: b = alpha * a
 */
template<typename T>
__global__ void scale_kernel(const T* a, T alpha, T* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = alpha * a[idx];
    }
}

/**
 * Fill kernel: x = value
 */
template<typename T>
__global__ void fill_kernel(T* x, T value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = value;
    }
}

// ============================================================================
// Vectorized Kernels (float4 for better memory bandwidth)
// ============================================================================

/**
 * Vectorized addition using float4 (4x memory coalescing)
 */
__global__ void add_kernel_vec4(const float4* a, const float4* b, float4* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 va = a[idx];
        float4 vb = b[idx];
        c[idx] = make_float4(
            va.x + vb.x,
            va.y + vb.y,
            va.z + vb.z,
            va.w + vb.w
        );
    }
}

/**
 * Vectorized multiplication using float4
 */
__global__ void mul_kernel_vec4(const float4* a, const float4* b, float4* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 va = a[idx];
        float4 vb = b[idx];
        c[idx] = make_float4(
            va.x * vb.x,
            va.y * vb.y,
            va.z * vb.z,
            va.w * vb.w
        );
    }
}

/**
 * Vectorized scale using float4
 */
__global__ void scale_kernel_vec4(const float4* a, float alpha, float4* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 va = a[idx];
        b[idx] = make_float4(
            alpha * va.x,
            alpha * va.y,
            alpha * va.z,
            alpha * va.w
        );
    }
}

// ============================================================================
// Transpose Kernel
// ============================================================================

/**
 * Matrix transpose kernel with shared memory tiling
 */
template<typename T, int TILE_DIM = 32, int BLOCK_ROWS = 8>
__global__ void transpose_kernel(const T* input, T* output, int rows, int cols) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile into shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Transpose tile indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed tile to output
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================================
// Implementation Functions
// ============================================================================

template<typename T>
void cuda_add(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C) {
    if (A.size() != B.size() || A.size() != C.size()) {
        throw std::runtime_error("cuda_add: size mismatch");
    }

    size_t n = A.size();
    auto config = KernelConfig::for_elementwise(n);

    add_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<const T*>(A.device_data()),
        static_cast<const T*>(B.device_data()),
        static_cast<T*>(C.device_data()),
        n
    );
    CUDA_CHECK_LAST();
}

// Specialization for float with vectorization
template<>
void cuda_add<float>(const CudaTensor<float>& A, const CudaTensor<float>& B,
                     CudaTensor<float>& C) {
    if (A.size() != B.size() || A.size() != C.size()) {
        throw std::runtime_error("cuda_add: size mismatch");
    }

    size_t n = A.size();

    // Use vectorized kernel when size is divisible by 4 and large enough
    if (n % 4 == 0 && n >= 1024) {
        size_t n4 = n / 4;
        dim3 block(256);
        dim3 grid(static_cast<unsigned int>((n4 + 255) / 256));

        add_kernel_vec4<<<grid, block>>>(
            reinterpret_cast<const float4*>(A.device_data()),
            reinterpret_cast<const float4*>(B.device_data()),
            reinterpret_cast<float4*>(C.device_data()),
            n4
        );
    } else {
        auto config = KernelConfig::for_elementwise(n);
        add_kernel<float><<<config.grid, config.block>>>(
            static_cast<const float*>(A.device_data()),
            static_cast<const float*>(B.device_data()),
            static_cast<float*>(C.device_data()),
            n
        );
    }
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_mul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C) {
    if (A.size() != B.size() || A.size() != C.size()) {
        throw std::runtime_error("cuda_mul: size mismatch");
    }

    size_t n = A.size();
    auto config = KernelConfig::for_elementwise(n);

    mul_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<const T*>(A.device_data()),
        static_cast<const T*>(B.device_data()),
        static_cast<T*>(C.device_data()),
        n
    );
    CUDA_CHECK_LAST();
}

// Specialization for float with vectorization
template<>
void cuda_mul<float>(const CudaTensor<float>& A, const CudaTensor<float>& B,
                     CudaTensor<float>& C) {
    if (A.size() != B.size() || A.size() != C.size()) {
        throw std::runtime_error("cuda_mul: size mismatch");
    }

    size_t n = A.size();

    if (n % 4 == 0 && n >= 1024) {
        size_t n4 = n / 4;
        dim3 block(256);
        dim3 grid(static_cast<unsigned int>((n4 + 255) / 256));

        mul_kernel_vec4<<<grid, block>>>(
            reinterpret_cast<const float4*>(A.device_data()),
            reinterpret_cast<const float4*>(B.device_data()),
            reinterpret_cast<float4*>(C.device_data()),
            n4
        );
    } else {
        auto config = KernelConfig::for_elementwise(n);
        mul_kernel<float><<<config.grid, config.block>>>(
            static_cast<const float*>(A.device_data()),
            static_cast<const float*>(B.device_data()),
            static_cast<float*>(C.device_data()),
            n
        );
    }
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_scale(const CudaTensor<T>& A, T alpha, CudaTensor<T>& B) {
    if (A.size() != B.size()) {
        throw std::runtime_error("cuda_scale: size mismatch");
    }

    size_t n = A.size();
    auto config = KernelConfig::for_elementwise(n);

    scale_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<const T*>(A.device_data()),
        alpha,
        static_cast<T*>(B.device_data()),
        n
    );
    CUDA_CHECK_LAST();
}

// Specialization for float with vectorization
template<>
void cuda_scale<float>(const CudaTensor<float>& A, float alpha, CudaTensor<float>& B) {
    if (A.size() != B.size()) {
        throw std::runtime_error("cuda_scale: size mismatch");
    }

    size_t n = A.size();

    if (n % 4 == 0 && n >= 1024) {
        size_t n4 = n / 4;
        dim3 block(256);
        dim3 grid(static_cast<unsigned int>((n4 + 255) / 256));

        scale_kernel_vec4<<<grid, block>>>(
            reinterpret_cast<const float4*>(A.device_data()),
            alpha,
            reinterpret_cast<float4*>(B.device_data()),
            n4
        );
    } else {
        auto config = KernelConfig::for_elementwise(n);
        scale_kernel<float><<<config.grid, config.block>>>(
            static_cast<const float*>(A.device_data()),
            alpha,
            static_cast<float*>(B.device_data()),
            n
        );
    }
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_fill(CudaTensor<T>& x, T value) {
    size_t n = x.size();
    auto config = KernelConfig::for_elementwise(n);

    fill_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<T*>(x.device_data()),
        value,
        n
    );
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_transpose(const CudaTensor<T>& src, CudaTensor<T>& dst) {
    if (src.ndim() != 2) {
        throw std::runtime_error("cuda_transpose: input must be 2D");
    }

    int rows = static_cast<int>(src.shape()[0]);
    int cols = static_cast<int>(src.shape()[1]);

    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);

    transpose_kernel<T, TILE_DIM, BLOCK_ROWS><<<grid, block>>>(
        static_cast<const T*>(src.device_data()),
        static_cast<T*>(dst.device_data()),
        rows,
        cols
    );
    CUDA_CHECK_LAST();
}

// ============================================================================
// Broadcast helpers (used by Layer::forward_cuda_dev)
// ============================================================================

// y is (B, O) row-major OR rank-1 (O,). bias is (O,). Adds bias[o] to every row.
template<typename T>
__global__ void add_bias_broadcast_kernel(T* y, const T* bias, size_t B, size_t O) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * O;
    if (idx < total) {
        size_t o = idx % O;
        y[idx] = y[idx] + bias[o];
    }
}

template<typename T>
__global__ void apply_mask_broadcast_kernel(T* y, const T* mask, size_t B, size_t O) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * O;
    if (idx < total) {
        size_t o = idx % O;
        y[idx] = y[idx] * mask[o];
    }
}

template<typename T>
void cuda_add_bias_broadcast(CudaTensor<T>& y, const CudaTensor<T>& bias) {
    if (bias.ndim() != 1) {
        throw std::runtime_error("cuda_add_bias_broadcast: bias must be 1D");
    }
    size_t O = bias.size();
    size_t total = y.size();
    if (total % O != 0) {
        throw std::runtime_error("cuda_add_bias_broadcast: y size not divisible by bias size");
    }
    size_t B = total / O;
    auto config = KernelConfig::for_elementwise(total);
    add_bias_broadcast_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<T*>(y.device_data()),
        static_cast<const T*>(bias.device_data()),
        B, O);
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_apply_mask_broadcast(CudaTensor<T>& y, const CudaTensor<T>& mask) {
    if (mask.ndim() != 1) {
        throw std::runtime_error("cuda_apply_mask_broadcast: mask must be 1D");
    }
    size_t O = mask.size();
    size_t total = y.size();
    if (total % O != 0) {
        throw std::runtime_error("cuda_apply_mask_broadcast: y size not divisible by mask size");
    }
    size_t B = total / O;
    auto config = KernelConfig::for_elementwise(total);
    apply_mask_broadcast_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<T*>(y.device_data()),
        static_cast<const T*>(mask.device_data()),
        B, O);
    CUDA_CHECK_LAST();
}

// Row-wise mask kernel: y[r, c] *= mask[r]. y is (R, C) row-major; mask is (R,).
template<typename T>
__global__ void apply_mask_rows_broadcast_kernel(T* y, const T* mask,
                                                  size_t R, size_t C) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = R * C;
    if (idx < total) {
        size_t r = idx / C;
        y[idx] = y[idx] * mask[r];
    }
}

template<typename T>
void cuda_apply_mask_rows_broadcast(CudaTensor<T>& y, const CudaTensor<T>& mask) {
    if (mask.ndim() != 1) {
        throw std::runtime_error("cuda_apply_mask_rows_broadcast: mask must be 1D");
    }
    size_t R = mask.size();
    size_t total = y.size();
    if (total % R != 0) {
        throw std::runtime_error(
            "cuda_apply_mask_rows_broadcast: y size not divisible by mask size");
    }
    size_t C = total / R;
    auto config = KernelConfig::for_elementwise(total);
    apply_mask_rows_broadcast_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<T*>(y.device_data()),
        static_cast<const T*>(mask.device_data()),
        R, C);
    CUDA_CHECK_LAST();
}

// AXPY: y = y + alpha * x. Same-size, element-wise.
template<typename T>
__global__ void axpy_kernel(T* y, T alpha, const T* x, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = y[idx] + alpha * x[idx];
    }
}

template<typename T>
void cuda_axpy(CudaTensor<T>& y, T alpha, const CudaTensor<T>& x) {
    if (y.size() != x.size()) {
        throw std::runtime_error("cuda_axpy: size mismatch");
    }
    size_t n = y.size();
    auto config = KernelConfig::for_elementwise(n);
    axpy_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<T*>(y.device_data()),
        alpha,
        static_cast<const T*>(x.device_data()),
        n);
    CUDA_CHECK_LAST();
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

// Note: float specializations are defined above
template void cuda_add<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
template void cuda_mul<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
template void cuda_scale<double>(const CudaTensor<double>&, double, CudaTensor<double>&);
template void cuda_fill<float>(CudaTensor<float>&, float);
template void cuda_fill<double>(CudaTensor<double>&, double);
template void cuda_transpose<float>(const CudaTensor<float>&, CudaTensor<float>&);
template void cuda_transpose<double>(const CudaTensor<double>&, CudaTensor<double>&);
template void cuda_add_bias_broadcast<float>(CudaTensor<float>&, const CudaTensor<float>&);
template void cuda_add_bias_broadcast<double>(CudaTensor<double>&, const CudaTensor<double>&);
template void cuda_apply_mask_broadcast<float>(CudaTensor<float>&, const CudaTensor<float>&);
template void cuda_apply_mask_broadcast<double>(CudaTensor<double>&, const CudaTensor<double>&);
template void cuda_apply_mask_rows_broadcast<float>(CudaTensor<float>&, const CudaTensor<float>&);
template void cuda_apply_mask_rows_broadcast<double>(CudaTensor<double>&, const CudaTensor<double>&);
template void cuda_axpy<float>(CudaTensor<float>&, float, const CudaTensor<float>&);
template void cuda_axpy<double>(CudaTensor<double>&, double, const CudaTensor<double>&);

} // namespace cuda
} // namespace dnn

#endif // DNN_ENABLE_CUDA
