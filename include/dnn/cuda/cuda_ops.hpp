#pragma once

/**
 * CUDA Operations - GPU-accelerated tensor operations.
 *
 * When DNN_ENABLE_CUDA is defined:
 * - Functions are implemented in .cu files using cuBLAS and custom kernels
 * - Full GPU acceleration is available
 *
 * When DNN_ENABLE_CUDA is NOT defined:
 * - Stub implementations throw runtime_error
 *
 * Implementation uses:
 * - cuBLAS for matrix operations (gemm, gemv, dot)
 * - Custom CUDA kernels for element-wise ops and activations
 */

#include "cuda_stubs.hpp"
#include "cuda_tensor.hpp"
#include <stdexcept>

namespace dnn {
namespace cuda {

// ============================================================================
// Handle Wrappers
// ============================================================================

/**
 * cuBLAS handle wrapper.
 * When CUDA is enabled, manages cuBLAS handle lifecycle.
 */
struct CublasHandle {
    void* handle = nullptr;

#ifdef DNN_ENABLE_CUDA
    CublasHandle();
    ~CublasHandle();
#else
    CublasHandle() {
        if (is_cuda_available()) {
            // Would call cublasCreate(&handle)
        }
    }
    ~CublasHandle() {
        if (handle) {
            // Would call cublasDestroy(handle)
        }
    }
#endif
};

/**
 * cuDNN handle wrapper (stub - not currently used).
 */
struct CudnnHandle {
    void* handle = nullptr;

    CudnnHandle() {
        if (is_cuda_available()) {
            // Would call cudnnCreate(&handle)
        }
    }

    ~CudnnHandle() {
        if (handle) {
            // Would call cudnnDestroy(handle)
        }
    }
};

/**
 * Global CUDA handles (lazy initialization).
 */
inline CublasHandle& get_cublas_handle() {
    static CublasHandle handle;
    return handle;
}

inline CudnnHandle& get_cudnn_handle() {
    static CudnnHandle handle;
    return handle;
}

/**
 * CUDA kernel launch configuration.
 */
struct LaunchConfig {
    int grid_size = 1;
    int block_size = 256;
    size_t shared_memory = 0;
    CudaStream stream = nullptr;
};

// ============================================================================
// Function Declarations / Stubs
// ============================================================================

#ifdef DNN_ENABLE_CUDA

// When CUDA is enabled, these are implemented in .cu files

// Matrix Operations (cuda_blas.cu)
template<typename T>
void cuda_gemm(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C,
               T alpha = T(1), T beta = T(0),
               bool transpose_A = false, bool transpose_B = false);

template<typename T>
void cuda_gemv(const CudaTensor<T>& A, const CudaTensor<T>& x, CudaTensor<T>& y,
               T alpha = T(1), T beta = T(0), bool transpose = false);

template<typename T>
T cuda_dot(const CudaTensor<T>& x, const CudaTensor<T>& y);

// Element-wise Operations (cuda_elementwise.cu)
template<typename T>
void cuda_add(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

template<typename T>
void cuda_mul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

template<typename T>
void cuda_scale(const CudaTensor<T>& A, T alpha, CudaTensor<T>& B);

template<typename T>
void cuda_fill(CudaTensor<T>& x, T value);

template<typename T>
void cuda_copy(const CudaTensor<T>& src, CudaTensor<T>& dst);

template<typename T>
void cuda_transpose(const CudaTensor<T>& src, CudaTensor<T>& dst);

// Broadcast helpers (cuda_elementwise.cu). Used by Layer::forward_cuda_dev
// to fuse the per-step "add bias" and "zero inactive output columns"
// operations onto the GPU instead of round-tripping through host memory.
//
// cuda_add_bias_broadcast(y, bias):
//   y is (B, O) or (O,) row-major. bias is (O,). Adds bias[o] to every row.
//   When y is rank-1, equivalent to cuda_add(y, bias, y).
//
// cuda_apply_mask_broadcast(y, mask):
//   y is (B, O) or (O,). mask is (O,). Multiplies y[b, o] *= mask[o].
//   Used to zero columns for soft-removed (inactive) neurons without a
//   host round-trip. Mask values are typically 0 or 1.
template<typename T>
void cuda_add_bias_broadcast(CudaTensor<T>& y, const CudaTensor<T>& bias);

template<typename T>
void cuda_apply_mask_broadcast(CudaTensor<T>& y, const CudaTensor<T>& mask);

// Row-wise mask: y is (R, C) row-major or rank-1 (R,). mask is (R,).
// Multiplies y[r, c] *= mask[r] (vs the column-wise cuda_apply_mask_broadcast).
// Used by the on-device optimizer to zero dW rows for inactive / non-trainable
// neurons before accumulation.
template<typename T>
void cuda_apply_mask_rows_broadcast(CudaTensor<T>& y, const CudaTensor<T>& mask);

// AXPY: y = y + alpha * x. Element-wise, x and y must have the same size.
// Used by the on-device SGD step in Layer::apply_gradients to update weights
// without round-tripping dW through host memory.
template<typename T>
void cuda_axpy(CudaTensor<T>& y, T alpha, const CudaTensor<T>& x);

// Activation Functions (cuda_activations.cu)
template<typename T>
void cuda_relu(const CudaTensor<T>& x, CudaTensor<T>& y);

template<typename T>
void cuda_relu_backward(const CudaTensor<T>& x, const CudaTensor<T>& dy, CudaTensor<T>& dx);

template<typename T>
void cuda_sigmoid(const CudaTensor<T>& x, CudaTensor<T>& y);

template<typename T>
void cuda_sigmoid_backward(const CudaTensor<T>& y, const CudaTensor<T>& dy, CudaTensor<T>& dx);

template<typename T>
void cuda_tanh(const CudaTensor<T>& x, CudaTensor<T>& y);

template<typename T>
void cuda_tanh_backward(const CudaTensor<T>& y, const CudaTensor<T>& dy, CudaTensor<T>& dx);

template<typename T>
void cuda_softmax(const CudaTensor<T>& x, CudaTensor<T>& y);

// Reduction Operations (cuda_reductions.cu)
template<typename T>
T cuda_sum(const CudaTensor<T>& x);

template<typename T>
void cuda_sum_axis(const CudaTensor<T>& x, CudaTensor<T>& y, int axis);

template<typename T>
T cuda_mean(const CudaTensor<T>& x);

template<typename T>
T cuda_max(const CudaTensor<T>& x);

template<typename T>
T cuda_min(const CudaTensor<T>& x);

// Optimizer helpers (cuda_optimizers.cu)
// Clamp every element of `gradients` to [-clip_value, +clip_value].
template<typename T>
void launch_gradient_clip(T* gradients, T clip_value, size_t n,
                          cudaStream_t stream = nullptr);

// Loss Functions (cuda_reductions.cu)
template<typename T>
T cuda_cross_entropy(const CudaTensor<T>& predictions, const CudaTensor<T>& targets);

template<typename T>
T cuda_mse(const CudaTensor<T>& predictions, const CudaTensor<T>& targets);

// Random (not yet implemented)
template<typename T>
void cuda_randn(CudaTensor<T>& x, T mean = T(0), T stddev = T(1));

#else // !DNN_ENABLE_CUDA

// Stub implementations when CUDA is not enabled

// Optimizer helpers (no-CUDA stub: no stream parameter since
// cudaStream_t is unavailable without the toolkit).
template<typename T>
void launch_gradient_clip(T* gradients, T clip_value, size_t n) {
    (void)gradients; (void)clip_value; (void)n;
    throw std::runtime_error("launch_gradient_clip: CUDA not enabled");
}

// Matrix Operations
template<typename T>
void cuda_gemm(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C,
               T alpha = T(1), T beta = T(0),
               bool transpose_A = false, bool transpose_B = false) {
    (void)A; (void)B; (void)C; (void)alpha; (void)beta;
    (void)transpose_A; (void)transpose_B;
    throw std::runtime_error("cuda_gemm: CUDA not enabled");
}

template<typename T>
void cuda_gemv(const CudaTensor<T>& A, const CudaTensor<T>& x, CudaTensor<T>& y,
               T alpha = T(1), T beta = T(0), bool transpose = false) {
    (void)A; (void)x; (void)y; (void)alpha; (void)beta; (void)transpose;
    throw std::runtime_error("cuda_gemv: CUDA not enabled");
}

template<typename T>
T cuda_dot(const CudaTensor<T>& x, const CudaTensor<T>& y) {
    (void)x; (void)y;
    throw std::runtime_error("cuda_dot: CUDA not enabled");
}

// Element-wise Operations
template<typename T>
void cuda_add(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C) {
    (void)A; (void)B; (void)C;
    throw std::runtime_error("cuda_add: CUDA not enabled");
}

template<typename T>
void cuda_mul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C) {
    (void)A; (void)B; (void)C;
    throw std::runtime_error("cuda_mul: CUDA not enabled");
}

template<typename T>
void cuda_scale(const CudaTensor<T>& A, T alpha, CudaTensor<T>& B) {
    (void)A; (void)alpha; (void)B;
    throw std::runtime_error("cuda_scale: CUDA not enabled");
}

template<typename T>
void cuda_fill(CudaTensor<T>& x, T value) {
    (void)x; (void)value;
    throw std::runtime_error("cuda_fill: CUDA not enabled");
}

template<typename T>
void cuda_copy(const CudaTensor<T>& src, CudaTensor<T>& dst) {
    (void)src; (void)dst;
    throw std::runtime_error("cuda_copy: CUDA not enabled");
}

template<typename T>
void cuda_transpose(const CudaTensor<T>& src, CudaTensor<T>& dst) {
    (void)src; (void)dst;
    throw std::runtime_error("cuda_transpose: CUDA not enabled");
}

template<typename T>
void cuda_add_bias_broadcast(CudaTensor<T>& y, const CudaTensor<T>& bias) {
    (void)y; (void)bias;
    throw std::runtime_error("cuda_add_bias_broadcast: CUDA not enabled");
}

template<typename T>
void cuda_apply_mask_broadcast(CudaTensor<T>& y, const CudaTensor<T>& mask) {
    (void)y; (void)mask;
    throw std::runtime_error("cuda_apply_mask_broadcast: CUDA not enabled");
}

template<typename T>
void cuda_apply_mask_rows_broadcast(CudaTensor<T>& y, const CudaTensor<T>& mask) {
    (void)y; (void)mask;
    throw std::runtime_error("cuda_apply_mask_rows_broadcast: CUDA not enabled");
}

template<typename T>
void cuda_axpy(CudaTensor<T>& y, T alpha, const CudaTensor<T>& x) {
    (void)y; (void)alpha; (void)x;
    throw std::runtime_error("cuda_axpy: CUDA not enabled");
}

// Activation Functions
template<typename T>
void cuda_relu(const CudaTensor<T>& x, CudaTensor<T>& y) {
    (void)x; (void)y;
    throw std::runtime_error("cuda_relu: CUDA not enabled");
}

template<typename T>
void cuda_relu_backward(const CudaTensor<T>& x, const CudaTensor<T>& dy, CudaTensor<T>& dx) {
    (void)x; (void)dy; (void)dx;
    throw std::runtime_error("cuda_relu_backward: CUDA not enabled");
}

template<typename T>
void cuda_sigmoid(const CudaTensor<T>& x, CudaTensor<T>& y) {
    (void)x; (void)y;
    throw std::runtime_error("cuda_sigmoid: CUDA not enabled");
}

template<typename T>
void cuda_sigmoid_backward(const CudaTensor<T>& y, const CudaTensor<T>& dy, CudaTensor<T>& dx) {
    (void)y; (void)dy; (void)dx;
    throw std::runtime_error("cuda_sigmoid_backward: CUDA not enabled");
}

template<typename T>
void cuda_tanh(const CudaTensor<T>& x, CudaTensor<T>& y) {
    (void)x; (void)y;
    throw std::runtime_error("cuda_tanh: CUDA not enabled");
}

template<typename T>
void cuda_tanh_backward(const CudaTensor<T>& y, const CudaTensor<T>& dy, CudaTensor<T>& dx) {
    (void)y; (void)dy; (void)dx;
    throw std::runtime_error("cuda_tanh_backward: CUDA not enabled");
}

template<typename T>
void cuda_softmax(const CudaTensor<T>& x, CudaTensor<T>& y) {
    (void)x; (void)y;
    throw std::runtime_error("cuda_softmax: CUDA not enabled");
}

// Reduction Operations
template<typename T>
T cuda_sum(const CudaTensor<T>& x) {
    (void)x;
    throw std::runtime_error("cuda_sum: CUDA not enabled");
}

template<typename T>
void cuda_sum_axis(const CudaTensor<T>& x, CudaTensor<T>& y, int axis) {
    (void)x; (void)y; (void)axis;
    throw std::runtime_error("cuda_sum_axis: CUDA not enabled");
}

template<typename T>
T cuda_mean(const CudaTensor<T>& x) {
    (void)x;
    throw std::runtime_error("cuda_mean: CUDA not enabled");
}

template<typename T>
T cuda_max(const CudaTensor<T>& x) {
    (void)x;
    throw std::runtime_error("cuda_max: CUDA not enabled");
}

template<typename T>
T cuda_min(const CudaTensor<T>& x) {
    (void)x;
    throw std::runtime_error("cuda_min: CUDA not enabled");
}

// Loss Functions
template<typename T>
T cuda_cross_entropy(const CudaTensor<T>& predictions, const CudaTensor<T>& targets) {
    (void)predictions; (void)targets;
    throw std::runtime_error("cuda_cross_entropy: CUDA not enabled");
}

template<typename T>
T cuda_mse(const CudaTensor<T>& predictions, const CudaTensor<T>& targets) {
    (void)predictions; (void)targets;
    throw std::runtime_error("cuda_mse: CUDA not enabled");
}

// Random
template<typename T>
void cuda_randn(CudaTensor<T>& x, T mean = T(0), T stddev = T(1)) {
    (void)x; (void)mean; (void)stddev;
    throw std::runtime_error("cuda_randn: CUDA not enabled");
}

#endif // DNN_ENABLE_CUDA

} // namespace cuda
} // namespace dnn
