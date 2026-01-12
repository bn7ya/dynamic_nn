#pragma once

/**
 * CUDA Interface - GPU acceleration support.
 *
 * When DNN_ENABLE_CUDA is defined:
 * - Functions are implemented in cuda_runtime.cu
 * - Full CUDA functionality is available
 *
 * When DNN_ENABLE_CUDA is NOT defined:
 * - Stub implementations are used
 * - Functions throw runtime_error or return defaults
 *
 * To enable CUDA, define DNN_ENABLE_CUDA before including this header
 * and link against the CUDA runtime and cuBLAS.
 */

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <string>

namespace dnn {
namespace cuda {

// ============================================================================
// Device Information Structure
// ============================================================================

/**
 * Device information structure.
 * Holds properties of a CUDA device.
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

// ============================================================================
// Memory Types and Enums
// ============================================================================

/**
 * Memory allocation type.
 */
enum class MemoryType {
    Device,      // GPU memory
    Pinned,      // Pinned host memory (faster transfers)
    Managed      // Unified memory (auto-migrates)
};

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
 * CUDA stream handle.
 */
using CudaStream = void*;

/**
 * CUDA event for timing.
 */
struct CudaEvent {
    void* handle = nullptr;
};

// ============================================================================
// Function Declarations / Stubs
// ============================================================================

#ifdef DNN_ENABLE_CUDA

// When CUDA is enabled, these are implemented in cuda_runtime.cu

/**
 * Check if CUDA is available (has working GPU).
 */
bool is_cuda_available();

/**
 * Get device information.
 */
DeviceInfo get_device_info(int device_id = 0);

/**
 * Get number of available CUDA devices.
 */
int get_device_count();

/**
 * Set current CUDA device.
 */
void set_device(int device_id);

/**
 * Synchronize current device.
 */
void device_synchronize();

/**
 * Allocate GPU memory.
 */
void* cuda_malloc(size_t bytes, MemoryType type = MemoryType::Device);

/**
 * Free GPU memory.
 */
void cuda_free(void* ptr);

/**
 * Copy memory between host and device.
 */
void cuda_memcpy(void* dst, const void* src, size_t bytes, MemcpyKind kind);

/**
 * Create CUDA stream.
 */
CudaStream create_stream();

/**
 * Destroy CUDA stream.
 */
void destroy_stream(CudaStream stream);

/**
 * Synchronize stream.
 */
void stream_synchronize(CudaStream stream);

/**
 * Check for CUDA errors.
 */
void check_cuda_error(const char* file, int line);

/**
 * Create CUDA event.
 */
CudaEvent create_event();

/**
 * Destroy CUDA event.
 */
void destroy_event(CudaEvent& event);

/**
 * Record event on stream.
 */
void record_event(CudaEvent& event, CudaStream stream = nullptr);

/**
 * Get elapsed time between two events (milliseconds).
 */
float elapsed_time(CudaEvent& start, CudaEvent& end);

#else // !DNN_ENABLE_CUDA

// Stub implementations when CUDA is not enabled

inline bool is_cuda_available() {
    return false;
}

inline DeviceInfo get_device_info(int device_id = 0) {
    (void)device_id;
    DeviceInfo info;
    info.device_id = -1;
    info.name = "CUDA not enabled - using CPU";
    return info;
}

inline int get_device_count() {
    return 0;
}

inline void set_device(int device_id) {
    (void)device_id;
    throw std::runtime_error("CUDA not enabled. Rebuild with DNN_ENABLE_CUDA to use GPU.");
}

inline void device_synchronize() {
    // No-op without CUDA
}

inline void* cuda_malloc(size_t bytes, MemoryType type = MemoryType::Device) {
    (void)bytes;
    (void)type;
    throw std::runtime_error("CUDA not enabled. Cannot allocate GPU memory.");
}

inline void cuda_free(void* ptr) {
    (void)ptr;
    throw std::runtime_error("CUDA not enabled. Cannot free GPU memory.");
}

inline void cuda_memcpy(void* dst, const void* src, size_t bytes, MemcpyKind kind) {
    (void)dst;
    (void)src;
    (void)bytes;
    (void)kind;
    throw std::runtime_error("CUDA not enabled. Cannot perform GPU memory copy.");
}

inline CudaStream create_stream() {
    throw std::runtime_error("CUDA not enabled. Cannot create stream.");
}

inline void destroy_stream(CudaStream stream) {
    (void)stream;
}

inline void stream_synchronize(CudaStream stream) {
    (void)stream;
}

inline void check_cuda_error(const char* file, int line) {
    (void)file;
    (void)line;
}

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

#endif // DNN_ENABLE_CUDA

// ============================================================================
// Error Checking Macro
// ============================================================================

#ifndef DNN_ENABLE_CUDA
#define CUDA_CHECK() ::dnn::cuda::check_cuda_error(__FILE__, __LINE__)
#endif

} // namespace cuda
} // namespace dnn
