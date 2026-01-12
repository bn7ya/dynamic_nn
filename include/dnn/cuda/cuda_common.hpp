#pragma once

/**
 * CUDA Common Utilities - Error checking, kernel configuration, device properties.
 *
 * This header provides common utilities for CUDA programming including:
 * - Error checking macros (CUDA_CHECK, CUBLAS_CHECK)
 * - Kernel launch configuration helpers
 * - Device properties caching
 */

#ifdef DNN_ENABLE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <sstream>
#include <algorithm>

namespace dnn {
namespace cuda {

/**
 * CUDA error checking macro with file/line information.
 * Throws std::runtime_error on CUDA errors.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::ostringstream oss; \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " - " << cudaGetErrorString(error); \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

/**
 * cuBLAS error checking macro with file/line information.
 * Throws std::runtime_error on cuBLAS errors.
 */
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::ostringstream oss; \
            oss << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                << " - status code: " << static_cast<int>(status); \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

/**
 * Check last CUDA error (for kernel launches).
 */
#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::ostringstream oss; \
            oss << "CUDA kernel error at " << __FILE__ << ":" << __LINE__ \
                << " - " << cudaGetErrorString(error); \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

/**
 * Kernel launch configuration helper.
 * Provides optimal grid/block sizing for different operation types.
 */
struct KernelConfig {
    dim3 grid;
    dim3 block;
    size_t shared_memory;
    cudaStream_t stream;

    KernelConfig()
        : grid(1), block(256), shared_memory(0), stream(nullptr) {}

    /**
     * Configuration for element-wise operations.
     * Uses 256 threads per block with minimal shared memory.
     */
    static KernelConfig for_elementwise(size_t n, cudaStream_t stream = nullptr) {
        KernelConfig config;
        config.block = dim3(256);
        config.grid = dim3(static_cast<unsigned int>((n + 255) / 256));
        config.shared_memory = 0;
        config.stream = stream;
        return config;
    }

    /**
     * Configuration for reduction operations.
     * Uses shared memory for partial reductions.
     */
    static KernelConfig for_reduction(size_t n, cudaStream_t stream = nullptr) {
        KernelConfig config;
        config.block = dim3(256);
        config.grid = dim3(static_cast<unsigned int>(
            std::min((n + 511) / 512, static_cast<size_t>(1024))));
        config.shared_memory = 256 * sizeof(float);
        config.stream = stream;
        return config;
    }

    /**
     * Configuration for matrix operations.
     * Uses 2D thread blocks for better memory coalescing.
     */
    static KernelConfig for_matrix(size_t rows, size_t cols, cudaStream_t stream = nullptr) {
        KernelConfig config;
        config.block = dim3(16, 16);
        config.grid = dim3(
            static_cast<unsigned int>((cols + 15) / 16),
            static_cast<unsigned int>((rows + 15) / 16));
        config.shared_memory = 0;
        config.stream = stream;
        return config;
    }
};

/**
 * Device properties cache.
 * Provides cached access to CUDA device properties to avoid repeated queries.
 */
struct DeviceProperties {
    int device_id;
    int compute_major;
    int compute_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_shared_memory_per_block;
    int warp_size;
    size_t total_global_memory;
    bool supports_tensor_cores;

    DeviceProperties()
        : device_id(-1)
        , compute_major(0)
        , compute_minor(0)
        , multiprocessor_count(0)
        , max_threads_per_block(0)
        , max_shared_memory_per_block(0)
        , warp_size(32)
        , total_global_memory(0)
        , supports_tensor_cores(false) {}

    /**
     * Get cached properties for current device.
     * Lazy initialization on first call.
     */
    static DeviceProperties& current() {
        static DeviceProperties props;
        static bool initialized = false;

        if (!initialized) {
            int device;
            if (cudaGetDevice(&device) == cudaSuccess) {
                cudaDeviceProp cuda_props;
                if (cudaGetDeviceProperties(&cuda_props, device) == cudaSuccess) {
                    props.device_id = device;
                    props.compute_major = cuda_props.major;
                    props.compute_minor = cuda_props.minor;
                    props.multiprocessor_count = cuda_props.multiProcessorCount;
                    props.max_threads_per_block = cuda_props.maxThreadsPerBlock;
                    props.max_shared_memory_per_block = static_cast<int>(cuda_props.sharedMemPerBlock);
                    props.warp_size = cuda_props.warpSize;
                    props.total_global_memory = cuda_props.totalGlobalMem;
                    // Tensor cores available on Volta (7.0) and later
                    props.supports_tensor_cores = (cuda_props.major >= 7);
                    initialized = true;
                }
            }
        }

        return props;
    }

    /**
     * Get compute capability as a single integer (e.g., 75 for 7.5).
     */
    int compute_capability() const {
        return compute_major * 10 + compute_minor;
    }
};

/**
 * RAII wrapper for CUDA streams.
 */
class ScopedStream {
public:
    ScopedStream() : stream_(nullptr), owns_(false) {}

    explicit ScopedStream(bool create) : stream_(nullptr), owns_(create) {
        if (create) {
            CUDA_CHECK(cudaStreamCreate(&stream_));
        }
    }

    ~ScopedStream() {
        if (owns_ && stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    // Move only
    ScopedStream(ScopedStream&& other) noexcept
        : stream_(other.stream_), owns_(other.owns_) {
        other.stream_ = nullptr;
        other.owns_ = false;
    }

    ScopedStream& operator=(ScopedStream&& other) noexcept {
        if (this != &other) {
            if (owns_ && stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            owns_ = other.owns_;
            other.stream_ = nullptr;
            other.owns_ = false;
        }
        return *this;
    }

    // No copy
    ScopedStream(const ScopedStream&) = delete;
    ScopedStream& operator=(const ScopedStream&) = delete;

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }

    void synchronize() {
        if (stream_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }

private:
    cudaStream_t stream_;
    bool owns_;
};

/**
 * RAII wrapper for CUDA events.
 */
class ScopedEvent {
public:
    ScopedEvent() : event_(nullptr) {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~ScopedEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }

    // Move only
    ScopedEvent(ScopedEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    ScopedEvent& operator=(ScopedEvent&& other) noexcept {
        if (this != &other) {
            if (event_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    // No copy
    ScopedEvent(const ScopedEvent&) = delete;
    ScopedEvent& operator=(const ScopedEvent&) = delete;

    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }

    void record(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    static float elapsed_ms(const ScopedEvent& start, const ScopedEvent& end) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
        return ms;
    }

private:
    cudaEvent_t event_;
};

} // namespace cuda
} // namespace dnn

#endif // DNN_ENABLE_CUDA
