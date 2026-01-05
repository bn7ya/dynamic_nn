#pragma once

/**
 * CUDA Stubs - Placeholder interfaces for future GPU acceleration.
 *
 * This file provides stub implementations that compile without CUDA.
 * When CUDA support is implemented, these stubs will be replaced with
 * actual CUDA implementations.
 *
 * To enable CUDA, define DNN_ENABLE_CUDA before including this header
 * and link against the CUDA runtime.
 */

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <string>

namespace dnn {
namespace cuda {

/**
 * CUDA availability check.
 */
inline bool is_cuda_available() {
#ifdef DNN_ENABLE_CUDA
    return true;  // Would actually check cudaGetDeviceCount
#else
    return false;
#endif
}

/**
 * Device information.
 */
struct DeviceInfo {
    int device_id = -1;
    std::string name = "No CUDA device";
    size_t total_memory = 0;
    size_t free_memory = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int multiprocessor_count = 0;
    int max_threads_per_block = 0;
    bool supports_tensor_cores = false;
};

/**
 * Get device information.
 */
inline DeviceInfo get_device_info(int device_id = 0) {
    (void)device_id;  // Suppress unused warning
    DeviceInfo info;
    info.device_id = -1;
    info.name = "CUDA not enabled - using CPU";
    return info;
}

/**
 * Get number of available CUDA devices.
 */
inline int get_device_count() {
    return 0;  // No devices when CUDA not enabled
}

/**
 * Set current CUDA device.
 */
inline void set_device(int device_id) {
    (void)device_id;
    throw std::runtime_error("CUDA not enabled. Rebuild with DNN_ENABLE_CUDA to use GPU.");
}

/**
 * Synchronize current device.
 */
inline void device_synchronize() {
    // No-op without CUDA
}

/**
 * Memory allocation type.
 */
enum class MemoryType {
    Device,      // GPU memory
    Pinned,      // Pinned host memory (faster transfers)
    Managed      // Unified memory (auto-migrates)
};

/**
 * CUDA memory allocation (stub).
 */
inline void* cuda_malloc(size_t bytes, MemoryType type = MemoryType::Device) {
    (void)bytes;
    (void)type;
    throw std::runtime_error("CUDA not enabled. Cannot allocate GPU memory.");
}

/**
 * CUDA memory free (stub).
 */
inline void cuda_free(void* ptr) {
    (void)ptr;
    throw std::runtime_error("CUDA not enabled. Cannot free GPU memory.");
}

/**
 * Memory copy direction.
 */
enum class MemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost
};

/**
 * CUDA memcpy (stub).
 */
inline void cuda_memcpy(void* dst, const void* src, size_t bytes, MemcpyKind kind) {
    (void)dst;
    (void)src;
    (void)bytes;
    (void)kind;
    throw std::runtime_error("CUDA not enabled. Cannot perform GPU memory copy.");
}

/**
 * CUDA stream handle (stub).
 */
using CudaStream = void*;

/**
 * Create CUDA stream (stub).
 */
inline CudaStream create_stream() {
    throw std::runtime_error("CUDA not enabled. Cannot create stream.");
}

/**
 * Destroy CUDA stream (stub).
 */
inline void destroy_stream(CudaStream stream) {
    (void)stream;
}

/**
 * Synchronize stream (stub).
 */
inline void stream_synchronize(CudaStream stream) {
    (void)stream;
}

/**
 * Check for CUDA errors (stub).
 */
inline void check_cuda_error(const char* file, int line) {
    (void)file;
    (void)line;
    // No-op without CUDA
}

#define CUDA_CHECK() ::dnn::cuda::check_cuda_error(__FILE__, __LINE__)

/**
 * CUDA event for timing (stub).
 */
struct CudaEvent {
    void* handle = nullptr;
};

inline CudaEvent create_event() {
    throw std::runtime_error("CUDA not enabled. Cannot create event.");
}

inline void destroy_event(CudaEvent& event) {
    (void)event;
}

inline void record_event(CudaEvent& event, CudaStream stream = nullptr) {
    (void)event;
    (void)stream;
}

inline float elapsed_time(CudaEvent& start, CudaEvent& end) {
    (void)start;
    (void)end;
    return 0.0f;
}

} // namespace cuda
} // namespace dnn
