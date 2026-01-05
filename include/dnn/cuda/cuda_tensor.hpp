#pragma once

/**
 * CUDA Tensor - GPU tensor stub for future implementation.
 *
 * Provides interface for GPU-accelerated tensor operations.
 * Currently implemented as stubs that throw when CUDA is not enabled.
 */

#include "cuda_stubs.hpp"
#include "../core/tensor.hpp"
#include <memory>
#include <stdexcept>

namespace dnn {
namespace cuda {

/**
 * GPU Tensor class (stub implementation).
 *
 * When CUDA is enabled, this class will manage GPU memory and
 * provide accelerated tensor operations.
 */
template<typename T = float>
class CudaTensor {
public:
    /**
     * Create empty tensor on GPU.
     */
    explicit CudaTensor(const std::vector<size_t>& shape)
        : shape_(shape)
        , size_(1)
        , device_ptr_(nullptr) {
        for (auto dim : shape) {
            size_ *= dim;
        }

        if (!is_cuda_available()) {
            throw std::runtime_error(
                "CudaTensor requires CUDA. Use core::Tensor for CPU operations."
            );
        }
    }

    /**
     * Create tensor from CPU tensor (copy to GPU).
     */
    explicit CudaTensor(const core::Tensor<T>& cpu_tensor)
        : CudaTensor(cpu_tensor.shape()) {
        to_device(cpu_tensor);
    }

    /**
     * Destructor - free GPU memory.
     */
    ~CudaTensor() {
        if (device_ptr_) {
            try {
                cuda_free(device_ptr_);
            } catch (...) {
                // Ignore errors during destruction
            }
        }
    }

    // Disable copy (GPU memory management)
    CudaTensor(const CudaTensor&) = delete;
    CudaTensor& operator=(const CudaTensor&) = delete;

    // Enable move
    CudaTensor(CudaTensor&& other) noexcept
        : shape_(std::move(other.shape_))
        , size_(other.size_)
        , device_ptr_(other.device_ptr_) {
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaTensor& operator=(CudaTensor&& other) noexcept {
        if (this != &other) {
            if (device_ptr_) {
                cuda_free(device_ptr_);
            }
            shape_ = std::move(other.shape_);
            size_ = other.size_;
            device_ptr_ = other.device_ptr_;
            other.device_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * Copy data from CPU tensor to GPU.
     */
    void to_device(const core::Tensor<T>& cpu_tensor) {
        if (!is_cuda_available()) {
            throw std::runtime_error("CUDA not available");
        }

        if (cpu_tensor.size() != size_) {
            throw std::runtime_error("Size mismatch in to_device");
        }

        if (!device_ptr_) {
            device_ptr_ = cuda_malloc(size_ * sizeof(T), MemoryType::Device);
        }

        cuda_memcpy(device_ptr_, cpu_tensor.data(),
                   size_ * sizeof(T), MemcpyKind::HostToDevice);
    }

    /**
     * Copy data from GPU to CPU tensor.
     */
    core::Tensor<T> to_host() const {
        if (!is_cuda_available() || !device_ptr_) {
            throw std::runtime_error("No GPU data to copy");
        }

        core::Tensor<T> cpu_tensor(shape_);
        cuda_memcpy(cpu_tensor.data(), device_ptr_,
                   size_ * sizeof(T), MemcpyKind::DeviceToHost);
        return cpu_tensor;
    }

    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    void* device_data() { return device_ptr_; }
    const void* device_data() const { return device_ptr_; }

    /**
     * Check if tensor has GPU memory allocated.
     */
    bool is_allocated() const { return device_ptr_ != nullptr; }

private:
    std::vector<size_t> shape_;
    size_t size_;
    void* device_ptr_;
};

/**
 * Convenience function to move tensor to GPU.
 */
template<typename T>
CudaTensor<T> to_cuda(const core::Tensor<T>& cpu_tensor) {
    return CudaTensor<T>(cpu_tensor);
}

/**
 * Convenience function to move tensor to CPU.
 */
template<typename T>
core::Tensor<T> to_cpu(const CudaTensor<T>& gpu_tensor) {
    return gpu_tensor.to_host();
}

} // namespace cuda
} // namespace dnn
