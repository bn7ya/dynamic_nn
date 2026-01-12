/**
 * CUDA Runtime Implementation
 *
 * Implements device management, memory operations, and stream/event handling.
 * Replaces stub functions in cuda_stubs.hpp when DNN_ENABLE_CUDA is defined.
 */

#ifdef DNN_ENABLE_CUDA

#include "dnn/cuda/cuda_common.hpp"
#include "dnn/cuda/cuda_stubs.hpp"
#include <cuda_runtime.h>
#include <sstream>

namespace dnn {
namespace cuda {

// ============================================================================
// Device Management
// ============================================================================

bool is_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

DeviceInfo get_device_info(int device_id) {
    DeviceInfo info;

    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        info.device_id = -1;
        info.name = "Error getting device properties";
        return info;
    }

    info.device_id = device_id;
    info.name = props.name;
    info.total_memory = props.totalGlobalMem;

    // Get free memory (need to set device first)
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(device_id);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    info.free_memory = free_mem;

    cudaSetDevice(current_device);

    info.compute_capability_major = props.major;
    info.compute_capability_minor = props.minor;
    info.multiprocessor_count = props.multiProcessorCount;
    info.max_threads_per_block = props.maxThreadsPerBlock;

    // Tensor cores available on Volta (7.0) and later
    info.supports_tensor_cores = (props.major >= 7);

    return info;
}

int get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

void set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

void device_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Memory Management
// ============================================================================

void* cuda_malloc(size_t bytes, MemoryType type) {
    void* ptr = nullptr;

    switch (type) {
        case MemoryType::Device:
            CUDA_CHECK(cudaMalloc(&ptr, bytes));
            break;
        case MemoryType::Pinned:
            CUDA_CHECK(cudaMallocHost(&ptr, bytes));
            break;
        case MemoryType::Managed:
            CUDA_CHECK(cudaMallocManaged(&ptr, bytes));
            break;
    }

    return ptr;
}

void cuda_free(void* ptr) {
    if (!ptr) return;

    // Try device free first
    cudaError_t err = cudaFree(ptr);
    if (err == cudaSuccess) return;

    // If that fails, try host free (for pinned memory)
    cudaGetLastError();  // Clear error
    err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        // Clear error and ignore - memory may have been allocated differently
        cudaGetLastError();
    }
}

void cuda_memcpy(void* dst, const void* src, size_t bytes, MemcpyKind kind) {
    cudaMemcpyKind cuda_kind;

    switch (kind) {
        case MemcpyKind::HostToDevice:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case MemcpyKind::DeviceToHost:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case MemcpyKind::DeviceToDevice:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        case MemcpyKind::HostToHost:
            cuda_kind = cudaMemcpyHostToHost;
            break;
        default:
            throw std::runtime_error("Unknown MemcpyKind");
    }

    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cuda_kind));
}

// ============================================================================
// Stream Management
// ============================================================================

CudaStream create_stream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return static_cast<CudaStream>(stream);
}

void destroy_stream(CudaStream stream) {
    if (stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }
}

void stream_synchronize(CudaStream stream) {
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    } else {
        // Synchronize default stream
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
}

// ============================================================================
// Event Management
// ============================================================================

CudaEvent create_event() {
    CudaEvent event;
    cudaEvent_t cuda_event;
    CUDA_CHECK(cudaEventCreate(&cuda_event));
    event.handle = cuda_event;
    return event;
}

void destroy_event(CudaEvent& event) {
    if (event.handle) {
        cudaEventDestroy(static_cast<cudaEvent_t>(event.handle));
        event.handle = nullptr;
    }
}

void record_event(CudaEvent& event, CudaStream stream) {
    CUDA_CHECK(cudaEventRecord(
        static_cast<cudaEvent_t>(event.handle),
        static_cast<cudaStream_t>(stream)
    ));
}

float elapsed_time(CudaEvent& start, CudaEvent& end) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &ms,
        static_cast<cudaEvent_t>(start.handle),
        static_cast<cudaEvent_t>(end.handle)
    ));
    return ms;
}

// ============================================================================
// Error Handling
// ============================================================================

void check_cuda_error(const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line
            << " - " << cudaGetErrorString(error);
        throw std::runtime_error(oss.str());
    }
}

} // namespace cuda
} // namespace dnn

#endif // DNN_ENABLE_CUDA
