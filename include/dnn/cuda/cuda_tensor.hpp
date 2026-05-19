#pragma once

/**
 * CUDA Tensor - GPU tensor for accelerated operations.
 *
 * When DNN_ENABLE_CUDA is defined:
 * - Uses GPU memory via CUDA memory pool for efficient allocation
 * - Provides data transfer between CPU and GPU
 *
 * When DNN_ENABLE_CUDA is NOT defined:
 * - Throws runtime_error when attempting to create a CudaTensor
 */

#include "cuda_stubs.hpp"
#include "../core/tensor.hpp"
#include <memory>
#include <stdexcept>

#ifdef DNN_ENABLE_CUDA
#include "cuda_memory_pool.hpp"
#endif

namespace dnn {
namespace cuda {

/**
 * GPU Tensor class.
 *
 * Manages GPU memory and provides data transfer operations.
 * Move-only semantics to prevent accidental GPU memory copies.
 */
template<typename T = float>
class CudaTensor {
public:
    /**
     * Create tensor on GPU with given shape.
     * Allocates GPU memory but does not initialize values.
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

        // Allocate GPU memory
        allocate();
    }

    /**
     * Create tensor from CPU tensor (copy to GPU).
     */
    explicit CudaTensor(const core::Tensor<T>& cpu_tensor)
        : shape_(cpu_tensor.shape())
        , size_(cpu_tensor.size())
        , device_ptr_(nullptr) {
        if (!is_cuda_available()) {
            throw std::runtime_error(
                "CudaTensor requires CUDA. Use core::Tensor for CPU operations."
            );
        }

        allocate();
        to_device(cpu_tensor);
    }

    /**
     * Destructor - free GPU memory.
     */
    ~CudaTensor() {
        deallocate();
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
            deallocate();
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
            allocate();
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

    /**
     * Copy data to existing CPU tensor.
     */
    void to_host(core::Tensor<T>& cpu_tensor) const {
        if (!is_cuda_available() || !device_ptr_) {
            throw std::runtime_error("No GPU data to copy");
        }

        if (cpu_tensor.size() != size_) {
            throw std::runtime_error("Size mismatch in to_host");
        }

        cuda_memcpy(cpu_tensor.data(), device_ptr_,
                   size_ * sizeof(T), MemcpyKind::DeviceToHost);
    }

    /**
     * Copy exactly `n` elements starting at element `offset` from device
     * memory into host buffer `dst`. Avoids D2H-ing the whole (B, O)
     * tensor just to read one row for per-node metric recording.
     */
    void copy_to_host_strided(T* dst, size_t offset, size_t n) const {
        if (!is_cuda_available() || !device_ptr_) {
            throw std::runtime_error("No GPU data to copy");
        }
        if (offset + n > size_) {
            throw std::runtime_error("copy_to_host_strided: out of range");
        }
        const char* src = static_cast<const char*>(device_ptr_) +
                          offset * sizeof(T);
        cuda_memcpy(dst, src, n * sizeof(T), MemcpyKind::DeviceToHost);
    }

    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    size_t bytes() const { return size_ * sizeof(T); }

    void* device_data() { return device_ptr_; }
    const void* device_data() const { return device_ptr_; }

    /**
     * Check if tensor has GPU memory allocated.
     */
    bool is_allocated() const { return device_ptr_ != nullptr; }

    /**
     * Reshape tensor (must preserve total size).
     */
    void reshape(const std::vector<size_t>& new_shape) {
        size_t new_size = 1;
        for (auto dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != size_) {
            throw std::runtime_error("Reshape size mismatch");
        }
        shape_ = new_shape;
    }

private:
    void allocate() {
        if (size_ == 0) return;

#ifdef DNN_ENABLE_CUDA
        // Use memory pool for efficient allocation
        device_ptr_ = CudaMemoryPool::instance().allocate(size_ * sizeof(T));
#else
        device_ptr_ = cuda_malloc(size_ * sizeof(T), MemoryType::Device);
#endif
    }

    void deallocate() {
        if (device_ptr_) {
            try {
#ifdef DNN_ENABLE_CUDA
                CudaMemoryPool::instance().deallocate(device_ptr_);
#else
                cuda_free(device_ptr_);
#endif
            } catch (...) {
                // Ignore errors during destruction
            }
            device_ptr_ = nullptr;
        }
    }

    std::vector<size_t> shape_;
    size_t size_;
#ifdef DNN_ENABLE_CUDA
    // Stream this tensor is bound to. nullptr (the default CUDA stream)
    // preserves legacy behaviour. Stage workers running on per-worker
    // streams set this so kernel launches and async copies don't
    // serialise on the default stream.
    cudaStream_t stream_ = nullptr;
public:
    /** Bind this tensor to a CUDA stream. */
    void set_stream(cudaStream_t stream) { stream_ = stream; }
    /** The stream this tensor's ops will be launched on. */
    cudaStream_t stream() const { return stream_; }
private:
#endif
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
