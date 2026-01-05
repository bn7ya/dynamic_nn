#pragma once

#include "../exceptions/dnn_exception.hpp"
#include <atomic>
#include <mutex>
#include <cstddef>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#include <sys/resource.h>
#include <fstream>
#endif

namespace dnn {
namespace memory {

/**
 * Resource usage statistics.
 */
struct ResourceUsage {
    size_t memory_used_bytes;       // Currently used by DNN
    size_t memory_available_bytes;  // System available memory
    double memory_usage_percent;    // DNN memory as % of system
    double cpu_usage_percent;       // CPU usage (approximate)
    size_t tensor_count;            // Number of active tensors
    size_t total_parameters;        // Total network parameters
};

/**
 * Resource limits configuration.
 */
struct ResourceLimits {
    size_t max_memory_bytes = 0;        // 0 = no limit
    double max_memory_percent = 90.0;   // Max % of system memory
    size_t max_parameters = 0;          // 0 = no limit
    size_t max_tensor_size = 0;         // 0 = no limit (single tensor)
};

/**
 * Monitors and enforces resource usage limits.
 * Thread-safe singleton.
 */
class ResourceMonitor {
public:
    static ResourceMonitor& instance() {
        static ResourceMonitor instance;
        return instance;
    }

    ResourceMonitor(const ResourceMonitor&) = delete;
    ResourceMonitor& operator=(const ResourceMonitor&) = delete;

    /**
     * Get current resource usage.
     */
    ResourceUsage current_usage() const {
        ResourceUsage usage;
        usage.memory_used_bytes = tensor_bytes_.load();
        usage.memory_available_bytes = get_available_memory();
        usage.tensor_count = tensor_count_.load();
        usage.total_parameters = total_parameters_.load();

        size_t total_mem = get_total_memory();
        if (total_mem > 0) {
            usage.memory_usage_percent =
                (static_cast<double>(usage.memory_used_bytes) / total_mem) * 100.0;
        } else {
            usage.memory_usage_percent = 0.0;
        }

        usage.cpu_usage_percent = 0.0; // Simplified - would need proper tracking

        return usage;
    }

    /**
     * Set resource limits.
     */
    void set_limits(const ResourceLimits& limits) {
        std::lock_guard<std::mutex> lock(mutex_);
        limits_ = limits;
    }

    /**
     * Get current limits.
     */
    ResourceLimits limits() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return limits_;
    }

    /**
     * Check if memory allocation is allowed.
     * @throws MemoryException if allocation would exceed limits
     */
    void check_memory_available(size_t required_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t current = tensor_bytes_.load();
        size_t new_total = current + required_bytes;

        // Check absolute limit
        if (limits_.max_memory_bytes > 0 && new_total > limits_.max_memory_bytes) {
            throw exceptions::MemoryException(
                required_bytes,
                limits_.max_memory_bytes - current,
                "Allocation would exceed max_memory_bytes limit"
            );
        }

        // Check percentage limit
        size_t total_mem = get_total_memory();
        if (total_mem > 0) {
            double new_percent = (static_cast<double>(new_total) / total_mem) * 100.0;
            if (new_percent > limits_.max_memory_percent) {
                size_t max_allowed = static_cast<size_t>(
                    (limits_.max_memory_percent / 100.0) * total_mem
                );
                throw exceptions::MemoryException(
                    required_bytes,
                    max_allowed > current ? max_allowed - current : 0,
                    "Allocation would exceed max_memory_percent limit"
                );
            }
        }

        // Check system available memory
        size_t available = get_available_memory();
        if (required_bytes > available) {
            throw exceptions::MemoryException(
                required_bytes,
                available,
                "Insufficient system memory available"
            );
        }
    }

    /**
     * Check if tensor allocation is allowed.
     * @throws MemoryException if tensor would exceed limits
     */
    void check_can_allocate_tensor(size_t tensor_bytes) {
        if (limits_.max_tensor_size > 0 && tensor_bytes > limits_.max_tensor_size) {
            throw exceptions::MemoryException(
                tensor_bytes,
                limits_.max_tensor_size,
                "Single tensor exceeds max_tensor_size limit"
            );
        }
        check_memory_available(tensor_bytes);
    }

    /**
     * Register tensor allocation.
     */
    void register_tensor(size_t bytes, size_t parameters = 0) {
        tensor_bytes_ += bytes;
        tensor_count_++;
        if (parameters > 0) {
            total_parameters_ += parameters;
        }
    }

    /**
     * Unregister tensor allocation.
     */
    void unregister_tensor(size_t bytes, size_t parameters = 0) {
        tensor_bytes_ -= bytes;
        tensor_count_--;
        if (parameters > 0) {
            total_parameters_ -= parameters;
        }
    }

    /**
     * Reset all tracking.
     */
    void reset() {
        tensor_bytes_ = 0;
        tensor_count_ = 0;
        total_parameters_ = 0;
    }

private:
    ResourceMonitor() = default;

    static size_t get_total_memory() {
#ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&memInfo)) {
            return static_cast<size_t>(memInfo.ullTotalPhys);
        }
        return 0;
#else
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
        }
        return 0;
#endif
    }

    static size_t get_available_memory() {
#ifdef _WIN32
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&memInfo)) {
            return static_cast<size_t>(memInfo.ullAvailPhys);
        }
        return 0;
#else
        long pages = sysconf(_SC_AVPHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
        }
        return 0;
#endif
    }

    mutable std::mutex mutex_;
    ResourceLimits limits_;
    std::atomic<size_t> tensor_bytes_{0};
    std::atomic<size_t> tensor_count_{0};
    std::atomic<size_t> total_parameters_{0};
};

} // namespace memory
} // namespace dnn
