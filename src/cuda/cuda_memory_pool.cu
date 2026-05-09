/**
 * CUDA Memory Pool Implementation
 *
 * Provides efficient GPU memory allocation through size-class pooling.
 * Reduces cudaMalloc/cudaFree overhead for repeated allocations.
 */

#ifdef DNN_ENABLE_CUDA

#include "dnn/cuda/cuda_memory_pool.hpp"
#include "dnn/cuda/cuda_common.hpp"
#include <stdexcept>
#include <algorithm>

namespace dnn {
namespace cuda {

// ============================================================================
// Singleton Instance
// ============================================================================

CudaMemoryPool& CudaMemoryPool::instance() {
    static CudaMemoryPool pool;
    return pool;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

CudaMemoryPool::CudaMemoryPool() {
    // Initialize pools with size classes
    // Each class is 2x the previous (256, 512, 1024, ... up to 256MB)
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        pools_[i].block_size = class_to_size(i);
        pools_[i].blocks_allocated = 0;
    }
}

CudaMemoryPool::~CudaMemoryPool() {
    try {
        clear();
    } catch (...) {
        // Ignore errors during destruction
    }
}

// ============================================================================
// Size Class Helpers
// ============================================================================

size_t CudaMemoryPool::size_to_class(size_t size) {
    if (size <= MIN_BLOCK_SIZE) return 0;

    size_t block_size = MIN_BLOCK_SIZE;
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        if (size <= block_size) return i;
        block_size *= 2;
    }

    return NUM_SIZE_CLASSES;  // Too large for pool
}

size_t CudaMemoryPool::class_to_size(size_t class_idx) {
    return MIN_BLOCK_SIZE << class_idx;
}

size_t CudaMemoryPool::align_size(size_t size) {
    // Align to 256 bytes for optimal GPU memory access
    return (size + 255) & ~static_cast<size_t>(255);
}

// ============================================================================
// Allocation Mode
// ============================================================================

void CudaMemoryPool::set_alloc_mode(AllocMode mode) {
    alloc_mode_.store(mode, std::memory_order_release);
}

CudaMemoryPool::AllocMode CudaMemoryPool::alloc_mode() const {
    return alloc_mode_.load(std::memory_order_acquire);
}

cudaError_t CudaMemoryPool::device_alloc(void** ptr, size_t bytes) {
    AllocMode mode = alloc_mode_.load(std::memory_order_acquire);
    if (mode == AllocMode::Managed) {
        // cudaMallocManaged returns a pointer that's valid on both device
        // and host; the CUDA driver pages between VRAM and host RAM as
        // needed. cudaFree handles both kinds of allocations, so the rest
        // of the pool's free path doesn't need to know which mode produced
        // a given block.
        return cudaMallocManaged(ptr, bytes, cudaMemAttachGlobal);
    }
    return cudaMalloc(ptr, bytes);
}

// ============================================================================
// Allocation Methods
// ============================================================================

void* CudaMemoryPool::allocate(size_t bytes, cudaStream_t stream) {
    (void)stream;  // Currently not using async allocation

    if (bytes == 0) return nullptr;

    // Align size
    bytes = align_size(bytes);

    // Use pool for small-to-medium allocations
    if (bytes <= MAX_POOLED_SIZE) {
        return allocate_from_pool(bytes);
    } else {
        return allocate_large(bytes);
    }
}

void* CudaMemoryPool::allocate_from_pool(size_t size) {
    size_t class_idx = size_to_class(size);

    if (class_idx >= NUM_SIZE_CLASSES) {
        return allocate_large(size);
    }

    size_t block_size = class_to_size(class_idx);

    std::lock_guard<std::mutex> lock(pool_mutex_);

    Pool& pool = pools_[class_idx];

    // Try to reuse from free list
    if (!pool.free_list.empty()) {
        void* ptr = pool.free_list.back();
        pool.free_list.pop_back();
        allocation_sizes_[ptr] = block_size;
        return ptr;
    }

    // Allocate new block via the mode-aware helper (cudaMalloc or
    // cudaMallocManaged depending on alloc_mode_).
    void* ptr = nullptr;
    cudaError_t err = device_alloc(&ptr, block_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    pool.blocks_allocated++;
    total_allocated_ += block_size;

    // Update peak usage
    size_t current = total_allocated_.load();
    size_t peak = peak_allocated_.load();
    while (current > peak && !peak_allocated_.compare_exchange_weak(peak, current)) {
        // CAS loop
    }

    allocation_sizes_[ptr] = block_size;
    return ptr;
}

void* CudaMemoryPool::allocate_large(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = device_alloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    std::lock_guard<std::mutex> lock(large_mutex_);
    large_allocations_[ptr] = size;
    allocation_sizes_[ptr] = size;
    total_allocated_ += size;

    // Update peak usage
    size_t current = total_allocated_.load();
    size_t peak = peak_allocated_.load();
    while (current > peak && !peak_allocated_.compare_exchange_weak(peak, current)) {
        // CAS loop
    }

    return ptr;
}

// ============================================================================
// Deallocation Methods
// ============================================================================

void CudaMemoryPool::deallocate(void* ptr) {
    if (!ptr) return;

    // Check if it's a large allocation
    {
        std::lock_guard<std::mutex> lock(large_mutex_);
        auto it = large_allocations_.find(ptr);
        if (it != large_allocations_.end()) {
            deallocate_large(ptr);
            return;
        }
    }

    // Return to pool
    std::lock_guard<std::mutex> lock(pool_mutex_);
    auto size_it = allocation_sizes_.find(ptr);
    if (size_it != allocation_sizes_.end()) {
        deallocate_to_pool(ptr, size_it->second);
    } else {
        // Unknown allocation - free directly
        cudaFree(ptr);
    }
}

void CudaMemoryPool::deallocate_to_pool(void* ptr, size_t size) {
    size_t class_idx = size_to_class(size);

    if (class_idx >= NUM_SIZE_CLASSES) {
        cudaFree(ptr);
        total_allocated_ -= size;
        allocation_sizes_.erase(ptr);
        return;
    }

    // Add to free list for reuse
    pools_[class_idx].free_list.push_back(ptr);
    // Note: Don't decrement total_allocated_ - memory is still allocated, just pooled
}

void CudaMemoryPool::deallocate_large(void* ptr) {
    auto it = large_allocations_.find(ptr);
    if (it != large_allocations_.end()) {
        size_t size = it->second;
        total_allocated_ -= size;
        large_allocations_.erase(it);
        allocation_sizes_.erase(ptr);
        cudaFree(ptr);
    }
}

// ============================================================================
// Async Methods
// ============================================================================

void* CudaMemoryPool::allocate_async(size_t bytes, cudaStream_t stream) {
    // For now, fall back to sync allocation
    // CUDA 11.2+ supports cudaMallocAsync for true async allocation
    return allocate(bytes, stream);
}

void CudaMemoryPool::deallocate_async(void* ptr, cudaStream_t stream) {
    (void)stream;
    deallocate(ptr);
}

// ============================================================================
// Pool Management
// ============================================================================

void CudaMemoryPool::reserve(size_t bytes) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Distribute across size classes
    size_t bytes_per_class = bytes / NUM_SIZE_CLASSES;

    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        size_t block_size = class_to_size(i);
        size_t blocks_to_add = std::max(bytes_per_class / block_size, static_cast<size_t>(2));

        for (size_t j = 0; j < blocks_to_add; ++j) {
            void* block = nullptr;
            cudaError_t err = device_alloc(&block, block_size);
            if (err == cudaSuccess) {
                pools_[i].free_list.push_back(block);
                pools_[i].blocks_allocated++;
                total_allocated_ += block_size;
                allocation_sizes_[block] = block_size;
            } else {
                // Stop if we can't allocate more
                break;
            }
        }
    }

    // Update peak
    size_t current = total_allocated_.load();
    size_t peak = peak_allocated_.load();
    while (current > peak && !peak_allocated_.compare_exchange_weak(peak, current)) {
        // CAS loop
    }
}

void CudaMemoryPool::clear() {
    std::lock_guard<std::mutex> pool_lock(pool_mutex_);
    std::lock_guard<std::mutex> large_lock(large_mutex_);

    // Free all pooled memory
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        for (void* ptr : pools_[i].free_list) {
            cudaFree(ptr);
        }
        total_allocated_ -= pools_[i].free_list.size() * pools_[i].block_size;
        pools_[i].free_list.clear();
        pools_[i].blocks_allocated = 0;
    }

    // Free large allocations
    for (auto& [ptr, size] : large_allocations_) {
        cudaFree(ptr);
        total_allocated_ -= size;
    }
    large_allocations_.clear();
    allocation_sizes_.clear();

    total_allocated_ = 0;
}

// ============================================================================
// Statistics
// ============================================================================

size_t CudaMemoryPool::bytes_allocated() const {
    return total_allocated_.load();
}

size_t CudaMemoryPool::bytes_in_pool() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    size_t total = 0;
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        total += pools_[i].free_list.size() * class_to_size(i);
    }
    return total;
}

size_t CudaMemoryPool::peak_usage() const {
    return peak_allocated_.load();
}

void CudaMemoryPool::reset_peak_usage() {
    peak_allocated_ = total_allocated_.load();
}

} // namespace cuda
} // namespace dnn

#endif // DNN_ENABLE_CUDA
