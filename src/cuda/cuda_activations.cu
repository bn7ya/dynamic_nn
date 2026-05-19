/**
 * CUDA Activation Functions
 *
 * Implements activation function kernels for neural networks.
 * Includes forward and backward passes for ReLU, Sigmoid, Tanh, and Softmax.
 */

#ifdef DNN_ENABLE_CUDA

#include "dnn/cuda/cuda_common.hpp"
#include "dnn/cuda/cuda_ops.hpp"
#include "dnn/cuda/cuda_tensor.hpp"
#include <cuda_runtime.h>
#include <cfloat>

namespace dnn {
namespace cuda {

// ============================================================================
// ReLU Kernels
// ============================================================================

/**
 * ReLU forward: y = max(0, x)
 */
template<typename T>
__global__ void relu_kernel(const T* x, T* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = x[idx];
        y[idx] = val > T(0) ? val : T(0);
    }
}

/**
 * ReLU backward: dx = dy * (x > 0)
 */
template<typename T>
__global__ void relu_backward_kernel(const T* x, const T* dy, T* dx, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dx[idx] = x[idx] > T(0) ? dy[idx] : T(0);
    }
}

// ============================================================================
// Sigmoid Kernels
// ============================================================================

/**
 * Sigmoid forward with numerical stability: y = 1 / (1 + exp(-x))
 *
 * For numerical stability:
 * - When x >= 0: y = 1 / (1 + exp(-x))
 * - When x < 0: y = exp(x) / (1 + exp(x))
 */
template<typename T>
__global__ void sigmoid_kernel(const T* x, T* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = x[idx];
        if (val >= T(0)) {
            y[idx] = T(1) / (T(1) + exp(-val));
        } else {
            T exp_val = exp(val);
            y[idx] = exp_val / (T(1) + exp_val);
        }
    }
}

/**
 * Sigmoid backward: dx = dy * y * (1 - y)
 */
template<typename T>
__global__ void sigmoid_backward_kernel(const T* y, const T* dy, T* dx, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T sigmoid_y = y[idx];
        dx[idx] = dy[idx] * sigmoid_y * (T(1) - sigmoid_y);
    }
}

// ============================================================================
// Tanh Kernels
// ============================================================================

/**
 * Tanh forward: y = tanh(x)
 */
template<typename T>
__global__ void tanh_kernel(const T* x, T* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = tanh(x[idx]);
    }
}

/**
 * Tanh backward: dx = dy * (1 - y^2)
 */
template<typename T>
__global__ void tanh_backward_kernel(const T* y, const T* dy, T* dx, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T tanh_y = y[idx];
        dx[idx] = dy[idx] * (T(1) - tanh_y * tanh_y);
    }
}

// ============================================================================
// Softmax Kernel
// ============================================================================

/**
 * Softmax forward with numerical stability (row-wise)
 *
 * Uses shared memory for reduction operations.
 * Two-pass algorithm:
 * 1. Find max for numerical stability
 * 2. Compute exp(x - max) and sum
 * 3. Normalize by sum
 */
template<typename T>
__global__ void softmax_kernel(const T* x, T* y, size_t batch_size, size_t dim) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    size_t batch_idx = blockIdx.x;
    size_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const T* x_row = x + batch_idx * dim;
    T* y_row = y + batch_idx * dim;

    // Pass 1: Find max (parallel reduction)
    T local_max = -FLT_MAX;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        local_max = max(local_max, x_row[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Reduce to find global max
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    T max_val = sdata[0];
    __syncthreads();

    // Pass 2: Compute exp(x - max) and sum
    T local_sum = T(0);
    for (size_t i = tid; i < dim; i += blockDim.x) {
        T exp_val = exp(x_row[i] - max_val);
        y_row[i] = exp_val;
        local_sum += exp_val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce to find global sum
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    T sum_val = sdata[0];
    __syncthreads();

    // Pass 3: Normalize
    T inv_sum = T(1) / sum_val;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        y_row[i] *= inv_sum;
    }
}

/**
 * Softmax for 1D tensor (single row)
 */
template<typename T>
__global__ void softmax_1d_kernel(const T* x, T* y, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    size_t tid = threadIdx.x;

    // Find max
    T local_max = -FLT_MAX;
    for (size_t i = tid; i < n; i += blockDim.x) {
        local_max = max(local_max, x[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    T max_val = sdata[0];
    __syncthreads();

    // Compute exp and sum
    T local_sum = T(0);
    for (size_t i = tid; i < n; i += blockDim.x) {
        T exp_val = exp(x[i] - max_val);
        y[i] = exp_val;
        local_sum += exp_val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    T sum_val = sdata[0];
    __syncthreads();

    // Normalize
    T inv_sum = T(1) / sum_val;
    for (size_t i = tid; i < n; i += blockDim.x) {
        y[i] *= inv_sum;
    }
}

/**
 * Softmax backward, batched (one row per block).
 * Jacobian-vector product: dx[i] = y[i] * (dy[i] - sum_j y[j]*dy[j]).
 */
template<typename T>
__global__ void softmax_backward_kernel(const T* y, const T* dy, T* dx,
                                         size_t batch_size, size_t dim) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    size_t batch_idx = blockIdx.x;
    size_t tid = threadIdx.x;
    if (batch_idx >= batch_size) return;

    const T* y_row = y + batch_idx * dim;
    const T* dy_row = dy + batch_idx * dim;
    T* dx_row = dx + batch_idx * dim;

    // Reduce dot = sum_j y[j] * dy[j].
    T local = T(0);
    for (size_t i = tid; i < dim; i += blockDim.x) {
        local += y_row[i] * dy_row[i];
    }
    sdata[tid] = local;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    T dot = sdata[0];
    __syncthreads();

    for (size_t i = tid; i < dim; i += blockDim.x) {
        dx_row[i] = y_row[i] * (dy_row[i] - dot);
    }
}

template<typename T>
__global__ void softmax_backward_1d_kernel(const T* y, const T* dy, T* dx,
                                            size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    size_t tid = threadIdx.x;

    T local = T(0);
    for (size_t i = tid; i < n; i += blockDim.x) local += y[i] * dy[i];
    sdata[tid] = local;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    T dot = sdata[0];
    __syncthreads();

    for (size_t i = tid; i < n; i += blockDim.x) {
        dx[i] = y[i] * (dy[i] - dot);
    }
}

// ============================================================================
// Implementation Functions
// ============================================================================

template<typename T>
void cuda_relu(const CudaTensor<T>& x, CudaTensor<T>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("cuda_relu: size mismatch");
    }

    size_t n = x.size();
    auto config = KernelConfig::for_elementwise(n);

    relu_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<const T*>(x.device_data()),
        static_cast<T*>(y.device_data()),
        n
    );
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_relu_backward(const CudaTensor<T>& x, const CudaTensor<T>& dy,
                        CudaTensor<T>& dx) {
    if (x.size() != dy.size() || x.size() != dx.size()) {
        throw std::runtime_error("cuda_relu_backward: size mismatch");
    }

    size_t n = x.size();
    auto config = KernelConfig::for_elementwise(n);

    relu_backward_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<const T*>(x.device_data()),
        static_cast<const T*>(dy.device_data()),
        static_cast<T*>(dx.device_data()),
        n
    );
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_sigmoid(const CudaTensor<T>& x, CudaTensor<T>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("cuda_sigmoid: size mismatch");
    }

    size_t n = x.size();
    auto config = KernelConfig::for_elementwise(n);

    sigmoid_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<const T*>(x.device_data()),
        static_cast<T*>(y.device_data()),
        n
    );
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_sigmoid_backward(const CudaTensor<T>& y, const CudaTensor<T>& dy,
                           CudaTensor<T>& dx) {
    if (y.size() != dy.size() || y.size() != dx.size()) {
        throw std::runtime_error("cuda_sigmoid_backward: size mismatch");
    }

    size_t n = y.size();
    auto config = KernelConfig::for_elementwise(n);

    sigmoid_backward_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<const T*>(y.device_data()),
        static_cast<const T*>(dy.device_data()),
        static_cast<T*>(dx.device_data()),
        n
    );
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_tanh(const CudaTensor<T>& x, CudaTensor<T>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("cuda_tanh: size mismatch");
    }

    size_t n = x.size();
    auto config = KernelConfig::for_elementwise(n);

    tanh_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<const T*>(x.device_data()),
        static_cast<T*>(y.device_data()),
        n
    );
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_tanh_backward(const CudaTensor<T>& y, const CudaTensor<T>& dy,
                        CudaTensor<T>& dx) {
    if (y.size() != dy.size() || y.size() != dx.size()) {
        throw std::runtime_error("cuda_tanh_backward: size mismatch");
    }

    size_t n = y.size();
    auto config = KernelConfig::for_elementwise(n);

    tanh_backward_kernel<T><<<config.grid, config.block, 0, config.stream>>>(
        static_cast<const T*>(y.device_data()),
        static_cast<const T*>(dy.device_data()),
        static_cast<T*>(dx.device_data()),
        n
    );
    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_softmax(const CudaTensor<T>& x, CudaTensor<T>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("cuda_softmax: size mismatch");
    }

    // Handle 1D and 2D cases
    if (x.ndim() == 1) {
        // Single row softmax
        size_t n = x.size();
        dim3 block(256);
        dim3 grid(1);
        size_t shared_mem = block.x * sizeof(T);

        softmax_1d_kernel<T><<<grid, block, shared_mem>>>(
            static_cast<const T*>(x.device_data()),
            static_cast<T*>(y.device_data()),
            n
        );
    } else if (x.ndim() == 2) {
        // Batched softmax (one softmax per row)
        size_t batch_size = x.shape()[0];
        size_t dim = x.shape()[1];

        dim3 block(256);
        dim3 grid(static_cast<unsigned int>(batch_size));
        size_t shared_mem = block.x * sizeof(T);

        softmax_kernel<T><<<grid, block, shared_mem>>>(
            static_cast<const T*>(x.device_data()),
            static_cast<T*>(y.device_data()),
            batch_size,
            dim
        );
    } else {
        throw std::runtime_error("cuda_softmax: only 1D and 2D tensors supported");
    }

    CUDA_CHECK_LAST();
}

template<typename T>
void cuda_softmax_backward(const CudaTensor<T>& y, const CudaTensor<T>& dy,
                           CudaTensor<T>& dx) {
    if (y.size() != dy.size() || y.size() != dx.size()) {
        throw std::runtime_error("cuda_softmax_backward: size mismatch");
    }

    if (y.ndim() == 1) {
        size_t n = y.size();
        dim3 block(256);
        dim3 grid(1);
        size_t shared_mem = block.x * sizeof(T);
        softmax_backward_1d_kernel<T><<<grid, block, shared_mem>>>(
            static_cast<const T*>(y.device_data()),
            static_cast<const T*>(dy.device_data()),
            static_cast<T*>(dx.device_data()),
            n
        );
    } else if (y.ndim() == 2) {
        size_t batch_size = y.shape()[0];
        size_t dim = y.shape()[1];
        dim3 block(256);
        dim3 grid(static_cast<unsigned int>(batch_size));
        size_t shared_mem = block.x * sizeof(T);
        softmax_backward_kernel<T><<<grid, block, shared_mem>>>(
            static_cast<const T*>(y.device_data()),
            static_cast<const T*>(dy.device_data()),
            static_cast<T*>(dx.device_data()),
            batch_size,
            dim
        );
    } else {
        throw std::runtime_error(
            "cuda_softmax_backward: only 1D and 2D tensors supported");
    }

    CUDA_CHECK_LAST();
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template void cuda_relu<float>(const CudaTensor<float>&, CudaTensor<float>&);
template void cuda_relu<double>(const CudaTensor<double>&, CudaTensor<double>&);
template void cuda_relu_backward<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
template void cuda_relu_backward<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);

template void cuda_sigmoid<float>(const CudaTensor<float>&, CudaTensor<float>&);
template void cuda_sigmoid<double>(const CudaTensor<double>&, CudaTensor<double>&);
template void cuda_sigmoid_backward<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
template void cuda_sigmoid_backward<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);

template void cuda_tanh<float>(const CudaTensor<float>&, CudaTensor<float>&);
template void cuda_tanh<double>(const CudaTensor<double>&, CudaTensor<double>&);
template void cuda_tanh_backward<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
template void cuda_tanh_backward<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);

template void cuda_softmax<float>(const CudaTensor<float>&, CudaTensor<float>&);
template void cuda_softmax<double>(const CudaTensor<double>&, CudaTensor<double>&);
template void cuda_softmax_backward<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
template void cuda_softmax_backward<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);

} // namespace cuda
} // namespace dnn

#endif // DNN_ENABLE_CUDA
