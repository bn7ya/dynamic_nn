/**
 * CUDA Reduction Operations
 *
 * Implements efficient parallel reduction kernels for sum, mean, max, min.
 * Uses warp-level primitives for optimal performance.
 */

#ifdef DNN_ENABLE_CUDA

#include "dnn/cuda/cuda_common.hpp"
#include "dnn/cuda/cuda_ops.hpp"
#include "dnn/cuda/cuda_tensor.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <limits>

namespace dnn {
namespace cuda {

// ============================================================================
// Warp-level Reduction Primitives
// ============================================================================

/**
 * Warp-level sum reduction using shuffle
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Warp-level max reduction using shuffle
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * Warp-level min reduction using shuffle
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_min(T val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ============================================================================
// Block-level Sum Reduction Kernel
// ============================================================================

/**
 * Two-stage parallel reduction for sum.
 * First reduces within blocks, then atomically adds to output.
 */
template<typename T>
__global__ void reduce_sum_kernel(const T* input, T* output, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load and add two elements per thread (better memory coalescing)
    T sum = T(0);
    if (idx < n) {
        sum = input[idx];
    }
    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction within block
    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp reduction (no sync needed within warp)
    if (tid < 32) {
        volatile T* vsdata = sdata;
        if (blockDim.x >= 64) vsdata[tid] += vsdata[tid + 32];
        if (blockDim.x >= 32) vsdata[tid] += vsdata[tid + 16];
        if (blockDim.x >= 16) vsdata[tid] += vsdata[tid + 8];
        if (blockDim.x >= 8) vsdata[tid] += vsdata[tid + 4];
        if (blockDim.x >= 4) vsdata[tid] += vsdata[tid + 2];
        if (blockDim.x >= 2) vsdata[tid] += vsdata[tid + 1];

        if (tid == 0) {
            atomicAdd(output, sdata[0]);
        }
    }
}

// ============================================================================
// Block-level Max Reduction Kernel
// ============================================================================

/**
 * Atomic max for float using compare-and-swap
 */
__device__ __forceinline__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    // NaN propagation (numpy semantics): any NaN poisons the result.
    if (isnan(val)) {
        atomicExch(address_as_int, __float_as_int(nanf("")));
        return;
    }
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (isnan(cur)) return;  // already NaN -> stays NaN
        float newv = val > cur ? val : cur;
        old = atomicCAS(address_as_int, assumed, __float_as_int(newv));
    } while (assumed != old);
}

/**
 * Atomic max for double using compare-and-swap
 */
__device__ __forceinline__ void atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(max(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

template<typename T>
__global__ void reduce_max_kernel(const T* input, T* output, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Initialize with smallest possible value
    T local_max = -FLT_MAX;
    if (idx < n) {
        local_max = input[idx];
    }
    if (idx + blockDim.x < n) {
        local_max = max(local_max, input[idx + blockDim.x]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Tree reduction
    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Final warp reduction
    if (tid < 32) {
        volatile T* vsdata = sdata;
        if (blockDim.x >= 64) vsdata[tid] = max(vsdata[tid], vsdata[tid + 32]);
        if (blockDim.x >= 32) vsdata[tid] = max(vsdata[tid], vsdata[tid + 16]);
        if (blockDim.x >= 16) vsdata[tid] = max(vsdata[tid], vsdata[tid + 8]);
        if (blockDim.x >= 8) vsdata[tid] = max(vsdata[tid], vsdata[tid + 4]);
        if (blockDim.x >= 4) vsdata[tid] = max(vsdata[tid], vsdata[tid + 2]);
        if (blockDim.x >= 2) vsdata[tid] = max(vsdata[tid], vsdata[tid + 1]);

        if (tid == 0) {
            atomicMaxFloat(output, sdata[0]);
        }
    }
}

// Double specialization
template<>
__global__ void reduce_max_kernel<double>(const double* input, double* output, size_t n) {
    extern __shared__ char shared_mem[];
    double* sdata = reinterpret_cast<double*>(shared_mem);

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double local_max = -DBL_MAX;
    if (idx < n) {
        local_max = input[idx];
    }
    if (idx + blockDim.x < n) {
        local_max = max(local_max, input[idx + blockDim.x]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile double* vsdata = sdata;
        if (blockDim.x >= 64) vsdata[tid] = max(vsdata[tid], vsdata[tid + 32]);
        if (blockDim.x >= 32) vsdata[tid] = max(vsdata[tid], vsdata[tid + 16]);
        if (blockDim.x >= 16) vsdata[tid] = max(vsdata[tid], vsdata[tid + 8]);
        if (blockDim.x >= 8) vsdata[tid] = max(vsdata[tid], vsdata[tid + 4]);
        if (blockDim.x >= 4) vsdata[tid] = max(vsdata[tid], vsdata[tid + 2]);
        if (blockDim.x >= 2) vsdata[tid] = max(vsdata[tid], vsdata[tid + 1]);

        if (tid == 0) {
            atomicMaxDouble(output, sdata[0]);
        }
    }
}

// ============================================================================
// Block-level Min Reduction Kernel
// ============================================================================

/**
 * Atomic min for float using compare-and-swap
 */
__device__ __forceinline__ void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    // NaN propagation (numpy semantics): any NaN poisons the result.
    if (isnan(val)) {
        atomicExch(address_as_int, __float_as_int(nanf("")));
        return;
    }
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (isnan(cur)) return;  // already NaN -> stays NaN
        float newv = val < cur ? val : cur;
        old = atomicCAS(address_as_int, assumed, __float_as_int(newv));
    } while (assumed != old);
}

/**
 * Atomic min for double using compare-and-swap
 */
__device__ __forceinline__ void atomicMinDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(min(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

template<typename T>
__global__ void reduce_min_kernel(const T* input, T* output, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T local_min = FLT_MAX;
    if (idx < n) {
        local_min = input[idx];
    }
    if (idx + blockDim.x < n) {
        local_min = min(local_min, input[idx + blockDim.x]);
    }

    sdata[tid] = local_min;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile T* vsdata = sdata;
        if (blockDim.x >= 64) vsdata[tid] = min(vsdata[tid], vsdata[tid + 32]);
        if (blockDim.x >= 32) vsdata[tid] = min(vsdata[tid], vsdata[tid + 16]);
        if (blockDim.x >= 16) vsdata[tid] = min(vsdata[tid], vsdata[tid + 8]);
        if (blockDim.x >= 8) vsdata[tid] = min(vsdata[tid], vsdata[tid + 4]);
        if (blockDim.x >= 4) vsdata[tid] = min(vsdata[tid], vsdata[tid + 2]);
        if (blockDim.x >= 2) vsdata[tid] = min(vsdata[tid], vsdata[tid + 1]);

        if (tid == 0) {
            atomicMinFloat(output, sdata[0]);
        }
    }
}

// Double specialization
template<>
__global__ void reduce_min_kernel<double>(const double* input, double* output, size_t n) {
    extern __shared__ char shared_mem[];
    double* sdata = reinterpret_cast<double*>(shared_mem);

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double local_min = DBL_MAX;
    if (idx < n) {
        local_min = input[idx];
    }
    if (idx + blockDim.x < n) {
        local_min = min(local_min, input[idx + blockDim.x]);
    }

    sdata[tid] = local_min;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile double* vsdata = sdata;
        if (blockDim.x >= 64) vsdata[tid] = min(vsdata[tid], vsdata[tid + 32]);
        if (blockDim.x >= 32) vsdata[tid] = min(vsdata[tid], vsdata[tid + 16]);
        if (blockDim.x >= 16) vsdata[tid] = min(vsdata[tid], vsdata[tid + 8]);
        if (blockDim.x >= 8) vsdata[tid] = min(vsdata[tid], vsdata[tid + 4]);
        if (blockDim.x >= 4) vsdata[tid] = min(vsdata[tid], vsdata[tid + 2]);
        if (blockDim.x >= 2) vsdata[tid] = min(vsdata[tid], vsdata[tid + 1]);

        if (tid == 0) {
            atomicMinDouble(output, sdata[0]);
        }
    }
}

// ============================================================================
// Sum Along Axis Kernel
// ============================================================================

template<typename T>
__global__ void sum_axis_kernel(const T* input, T* output,
                                size_t outer_size, size_t reduce_size, size_t inner_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer_size * inner_size;

    if (idx < total) {
        size_t outer = idx / inner_size;
        size_t inner = idx % inner_size;

        T sum = T(0);
        for (size_t r = 0; r < reduce_size; ++r) {
            sum += input[outer * reduce_size * inner_size + r * inner_size + inner];
        }
        output[idx] = sum;
    }
}

// ============================================================================
// Implementation Functions
// ============================================================================

template<typename T>
T cuda_sum(const CudaTensor<T>& x) {
    size_t n = x.size();

    // Allocate output on device
    T* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(T)));

    dim3 block(256);
    dim3 grid(static_cast<unsigned int>((n + block.x * 2 - 1) / (block.x * 2)));
    grid.x = min(grid.x, 1024u);  // Limit grid size
    size_t shared_mem = block.x * sizeof(T);

    reduce_sum_kernel<T><<<grid, block, shared_mem>>>(
        static_cast<const T*>(x.device_data()),
        d_result,
        n
    );
    CUDA_CHECK_LAST();

    T result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

template<typename T>
T cuda_mean(const CudaTensor<T>& x) {
    return cuda_sum(x) / static_cast<T>(x.size());
}

template<typename T>
T cuda_max(const CudaTensor<T>& x) {
    size_t n = x.size();

    T* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));

    // Initialize to smallest value
    T init_val = std::is_same<T, float>::value ? -FLT_MAX : -DBL_MAX;
    CUDA_CHECK(cudaMemcpy(d_result, &init_val, sizeof(T), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(static_cast<unsigned int>((n + block.x * 2 - 1) / (block.x * 2)));
    grid.x = min(grid.x, 1024u);
    size_t shared_mem = block.x * sizeof(T);

    reduce_max_kernel<T><<<grid, block, shared_mem>>>(
        static_cast<const T*>(x.device_data()),
        d_result,
        n
    );
    CUDA_CHECK_LAST();

    T result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

template<typename T>
T cuda_min(const CudaTensor<T>& x) {
    size_t n = x.size();

    T* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));

    // Initialize to largest value
    T init_val = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
    CUDA_CHECK(cudaMemcpy(d_result, &init_val, sizeof(T), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(static_cast<unsigned int>((n + block.x * 2 - 1) / (block.x * 2)));
    grid.x = min(grid.x, 1024u);
    size_t shared_mem = block.x * sizeof(T);

    reduce_min_kernel<T><<<grid, block, shared_mem>>>(
        static_cast<const T*>(x.device_data()),
        d_result,
        n
    );
    CUDA_CHECK_LAST();

    T result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

template<typename T>
void cuda_sum_axis(const CudaTensor<T>& x, CudaTensor<T>& y, int axis) {
    if (axis < 0 || static_cast<size_t>(axis) >= x.ndim()) {
        throw std::runtime_error("cuda_sum_axis: axis out of range");
    }

    // Calculate dimensions
    size_t outer_size = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        outer_size *= x.shape()[i];
    }

    size_t reduce_size = x.shape()[axis];

    size_t inner_size = 1;
    for (size_t i = axis + 1; i < x.ndim(); ++i) {
        inner_size *= x.shape()[i];
    }

    size_t output_size = outer_size * inner_size;
    auto config = KernelConfig::for_elementwise(output_size);

    sum_axis_kernel<T><<<config.grid, config.block>>>(
        static_cast<const T*>(x.device_data()),
        static_cast<T*>(y.device_data()),
        outer_size,
        reduce_size,
        inner_size
    );
    CUDA_CHECK_LAST();
}

// ============================================================================
// Loss Functions
// ============================================================================

template<typename T>
__global__ void cross_entropy_kernel(const T* predictions, const T* targets,
                                     T* result, size_t batch_size, size_t num_classes) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    size_t tid = threadIdx.x;
    size_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    T local_loss = T(0);
    if (batch_idx < batch_size) {
        const T* pred_row = predictions + batch_idx * num_classes;
        const T* target_row = targets + batch_idx * num_classes;

        for (size_t c = 0; c < num_classes; ++c) {
            // Clip predictions for numerical stability
            T pred = max(min(pred_row[c], T(1) - T(1e-7)), T(1e-7));
            local_loss -= target_row[c] * log(pred);
        }
    }

    sdata[tid] = local_loss;
    __syncthreads();

    // Reduce within block
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

template<typename T>
__global__ void mse_kernel(const T* predictions, const T* targets,
                           T* result, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    T local_mse = T(0);
    if (idx < n) {
        T diff = predictions[idx] - targets[idx];
        local_mse = diff * diff;
    }
    if (idx + blockDim.x < n) {
        T diff = predictions[idx + blockDim.x] - targets[idx + blockDim.x];
        local_mse += diff * diff;
    }

    sdata[tid] = local_mse;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

template<typename T>
T cuda_cross_entropy(const CudaTensor<T>& predictions, const CudaTensor<T>& targets) {
    if (predictions.ndim() != 2 || targets.ndim() != 2) {
        throw std::runtime_error("cuda_cross_entropy: inputs must be 2D");
    }

    size_t batch_size = predictions.shape()[0];
    size_t num_classes = predictions.shape()[1];

    T* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(T)));

    dim3 block(256);
    dim3 grid(static_cast<unsigned int>((batch_size + 255) / 256));
    size_t shared_mem = block.x * sizeof(T);

    cross_entropy_kernel<T><<<grid, block, shared_mem>>>(
        static_cast<const T*>(predictions.device_data()),
        static_cast<const T*>(targets.device_data()),
        d_result,
        batch_size,
        num_classes
    );
    CUDA_CHECK_LAST();

    T result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    return result / static_cast<T>(batch_size);
}

template<typename T>
T cuda_mse(const CudaTensor<T>& predictions, const CudaTensor<T>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("cuda_mse: size mismatch");
    }

    size_t n = predictions.size();

    T* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(T)));

    dim3 block(256);
    dim3 grid(static_cast<unsigned int>((n + block.x * 2 - 1) / (block.x * 2)));
    grid.x = min(grid.x, 1024u);
    size_t shared_mem = block.x * sizeof(T);

    mse_kernel<T><<<grid, block, shared_mem>>>(
        static_cast<const T*>(predictions.device_data()),
        static_cast<const T*>(targets.device_data()),
        d_result,
        n
    );
    CUDA_CHECK_LAST();

    T result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    return result / static_cast<T>(n);
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template float cuda_sum<float>(const CudaTensor<float>&);
template double cuda_sum<double>(const CudaTensor<double>&);
template float cuda_mean<float>(const CudaTensor<float>&);
template double cuda_mean<double>(const CudaTensor<double>&);
template float cuda_max<float>(const CudaTensor<float>&);
template double cuda_max<double>(const CudaTensor<double>&);
template float cuda_min<float>(const CudaTensor<float>&);
template double cuda_min<double>(const CudaTensor<double>&);
template void cuda_sum_axis<float>(const CudaTensor<float>&, CudaTensor<float>&, int);
template void cuda_sum_axis<double>(const CudaTensor<double>&, CudaTensor<double>&, int);
template float cuda_cross_entropy<float>(const CudaTensor<float>&, const CudaTensor<float>&);
template double cuda_cross_entropy<double>(const CudaTensor<double>&, const CudaTensor<double>&);
template float cuda_mse<float>(const CudaTensor<float>&, const CudaTensor<float>&);
template double cuda_mse<double>(const CudaTensor<double>&, const CudaTensor<double>&);

} // namespace cuda
} // namespace dnn

#endif // DNN_ENABLE_CUDA
