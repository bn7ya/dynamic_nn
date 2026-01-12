#pragma once

/**
 * CUDA Memory Pool - Efficient GPU memory allocation with pooling.
 *
 * Provides a memory pool for GPU allocations to reduce cudaMalloc overhead.
 * Uses size classes similar to standard memory allocators.
 */

#ifdef DNN_ENABLE_CUDA

#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <cstddef>

namespace dnn {
namespace cuda {

/**
 * GPU memory pool for efficient allocation/deallocation.
 *
 * Features:
 * - Size-class based pooling (256B to 256MB)
 * - Thread-safe operations
 * - Allocation tracking for statistics
 * - Automatic cleanup on destruction
 */
class CudaMemoryPool {
public:
    // Pool configuration
    static constexpr size_t NUM_SIZE_CLASSES = 20;
    static constexpr size_t MIN_BLOCK_SIZE = 256;           // 256 bytes
    static constexpr size_t MAX_POOLED_SIZE = 256 * 1024 * 1024;  // 256MB

    /**
     * Get singleton instance.
     */
    static CudaMemoryPool& instance();

    /**
     * Allocate GPU memory.
     * Returns memory from pool if available, otherwise allocates new.
     *
     * @param bytes Number of bytes to allocate
     * @param stream CUDA stream for async operations (optional)
     * @return Pointer to allocated GPU memory
     */
    void* allocate(size_t bytes, cudaStream_t stream = nullptr);

    /**
     * Deallocate GPU memory.
     * Returns memory to pool for reuse.
     *
     * @param ptr Pointer to GPU memory
     */
    void deallocate(void* ptr);

    /**
     * Async memory allocation.
     * Uses CUDA stream for async allocation.
     *
     * @param bytes Number of bytes to allocate
     * @param stream CUDA stream
     * @return Pointer to allocated GPU memory
     */
    void* allocate_async(size_t bytes, cudaStream_t stream);

    /**
     * Async memory deallocation.
     * Uses CUDA stream for async deallocation.
     *
     * @param ptr Pointer to GPU memory
     * @param stream CUDA stream
     */
    void deallocate_async(void* ptr, cudaStream_t stream);

    /**
     * Pre-allocate pool memory.
     * Useful for reducing allocation latency during critical operations.
     *
     * @param bytes Total bytes to pre-allocate (distributed across size classes)
     */
    void reserve(size_t bytes);

    /**
     * Release all pooled memory.
     * Frees all memory in the pool back to CUDA.
     */
    void clear();

    /**
     * Get total bytes currently allocated from CUDA.
     */
    size_t bytes_allocated() const;

    /**
     * Get bytes currently available in the pool.
     */
    size_t bytes_in_pool() const;

    /**
     * Get peak memory usage since pool creation.
     */
    size_t peak_usage() const;

    /**
     * Reset peak usage counter.
     */
    void reset_peak_usage();

private:
    CudaMemoryPool();
    ~CudaMemoryPool();

    // Non-copyable, non-movable
    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
    CudaMemoryPool(CudaMemoryPool&&) = delete;
    CudaMemoryPool& operator=(CudaMemoryPool&&) = delete;

    /**
     * Pool for a specific size class.
     */
    struct Pool {
        std::vector<void*> free_list;
        size_t block_size;
        size_t blocks_allocated;

        Pool() : block_size(0), blocks_allocated(0) {}
    };

    /**
     * Convert size to size class index.
     */
    static size_t size_to_class(size_t size);

    /**
     * Convert size class index to block size.
     */
    static size_t class_to_size(size_t class_idx);

    /**
     * Round size up to alignment boundary.
     */
    static size_t align_size(size_t size);

    // Internal allocation methods
    void* allocate_from_pool(size_t size);
    void deallocate_to_pool(void* ptr, size_t size);
    void* allocate_large(size_t size);
    void deallocate_large(void* ptr);

    // Thread safety
    mutable std::mutex pool_mutex_;
    mutable std::mutex large_mutex_;

    // Pools for each size class
    Pool pools_[NUM_SIZE_CLASSES];

    // Large allocations (not pooled)
    std::unordered_map<void*, size_t> large_allocations_;

    // Track allocation sizes for deallocation
    std::unordered_map<void*, size_t> allocation_sizes_;

    // Statistics
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_allocated_{0};
};

/**
 * RAII wrapper for pooled GPU memory.
 */
template<typename T>
class PooledMemory {
public:
    explicit PooledMemory(size_t count)
        : ptr_(nullptr), count_(count) {
        if (count > 0) {
            ptr_ = static_cast<T*>(CudaMemoryPool::instance().allocate(count * sizeof(T)));
        }
    }

    ~PooledMemory() {
        if (ptr_) {
            CudaMemoryPool::instance().deallocate(ptr_);
        }
    }

    // Move only
    PooledMemory(PooledMemory&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PooledMemory& operator=(PooledMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                CudaMemoryPool::instance().deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // No copy
    PooledMemory(const PooledMemory&) = delete;
    PooledMemory& operator=(const PooledMemory&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_;
    size_t count_;
};

} // namespace cuda
} // namespace dnn

#endif // DNN_ENABLE_CUDA
