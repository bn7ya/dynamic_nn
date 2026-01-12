#pragma once

#include <string>
#include <stdexcept>

namespace dnn {
namespace core {

/**
 * Device types for tensor computation.
 */
enum class Device {
    CPU,    // CPU computation (default)
    CUDA    // NVIDIA GPU via CUDA
};

/**
 * Get device name as string.
 */
inline std::string device_name(Device device) {
    switch (device) {
        case Device::CPU: return "cpu";
        case Device::CUDA: return "cuda";
        default: return "unknown";
    }
}

/**
 * Parse device from string.
 */
inline Device parse_device(const std::string& name) {
    if (name == "cpu" || name == "CPU") {
        return Device::CPU;
    } else if (name == "cuda" || name == "CUDA" || name == "gpu" || name == "GPU") {
        return Device::CUDA;
    }
    throw std::invalid_argument("Unknown device: " + name);
}

/**
 * Check if CUDA device is available at runtime.
 * Note: Actual check is done via cuda::is_cuda_available() in cuda_stubs.hpp
 */
inline bool cuda_available() {
#ifdef DNN_ENABLE_CUDA
    return true;  // Compile-time check; runtime check done elsewhere
#else
    return false;
#endif
}

} // namespace core
} // namespace dnn
