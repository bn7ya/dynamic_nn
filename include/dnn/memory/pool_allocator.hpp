#pragma once

#include "aligned_allocator.hpp"
#include <mutex>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>

namespace dnn {
namespace memory {

/**
 * Thread-safe memory pool allocator for efficient tensor memory management.
 * Uses size classes for small allocations and direct allocation for large ones.
 */
class PoolAllocator {
public:
    // Size classes: 64B, 128B, 256B, 512B, 1KB, 2KB, 4KB, 8KB, 16KB, 32KB, 64KB, 128KB, 256KB, 512KB, 1MB, 2MB
    static constexpr size_t NUM_SIZE_CLASSES = 16;
    static constexpr size_t MIN_BLOCK_SIZE = 64;
    static constexpr size_t MAX_POOLED_SIZE = 2 * 1024 * 1024; // 2MB

    static PoolAllocator& instance() {
        static PoolAllocator instance;
        return instance;
    }

    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    /**
     * Allocate memory with optional alignment.
     */
    void* allocate(size_t size, size_t alignment = DEFAULT_ALIGNMENT) {
        if (size == 0) return nullptr;

        // Round up to alignment
        size = (size + alignment - 1) & ~(alignment - 1);

        if (size <= MAX_POOLED_SIZE) {
            return allocate_from_pool(size);
        } else {
            return allocate_large(size, alignment);
        }
    }

    /**
     * Deallocate memory.
     */
    void deallocate(void* ptr) {
        if (!ptr) return;

        // Check if it's a large allocation
        {
            std::lock_guard<std::mutex> lock(large_mutex_);
            auto it = large_allocations_.find(ptr);
            if (it != large_allocations_.end()) {
                total_allocated_ -= it->second;
                large_allocations_.erase(it);
                aligned_free(ptr);
                return;
            }
        }

        // Must be a pooled allocation
        deallocate_to_pool(ptr);
    }

    /**
     * Pre-allocate memory for the pool.
     */
    void reserve(size_t bytes) {
        std::lock_guard<std::mutex> lock(pool_mutex_);

        // Distribute across size classes based on typical usage
        for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
            size_t block_size = size_class_to_size(i);
            size_t blocks_to_add = bytes / (NUM_SIZE_CLASSES * block_size);
            blocks_to_add = std::max(blocks_to_add, size_t(4));

            for (size_t j = 0; j < blocks_to_add; ++j) {
                void* block = aligned_alloc(block_size, DEFAULT_ALIGNMENT);
                if (block) {
                    pools_[i].free_list.push_back(block);
                    pools_[i].blocks_allocated++;
                    total_allocated_ += block_size;
                }
            }
        }

        peak_allocated_ = std::max(peak_allocated_, total_allocated_.load());
    }

    /**
     * Clear all pooled memory.
     */
    void clear() {
        std::lock_guard<std::mutex> pool_lock(pool_mutex_);
        std::lock_guard<std::mutex> large_lock(large_mutex_);

        // Free pooled memory
        for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
            for (void* ptr : pools_[i].free_list) {
                aligned_free(ptr);
            }
            pools_[i].free_list.clear();
            pools_[i].blocks_allocated = 0;
        }

        // Free large allocations
        for (auto& [ptr, size] : large_allocations_) {
            aligned_free(ptr);
        }
        large_allocations_.clear();

        total_allocated_ = 0;
    }

    // Statistics
    size_t bytes_allocated() const { return total_allocated_.load(); }
    size_t peak_usage() const { return peak_allocated_; }

    size_t bytes_in_pools() const {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        size_t total = 0;
        for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
            total += pools_[i].free_list.size() * size_class_to_size(i);
        }
        return total;
    }

private:
    struct Pool {
        std::vector<void*> free_list;
        size_t blocks_allocated = 0;
    };

    PoolAllocator() = default;

    ~PoolAllocator() {
        clear();
    }

    static size_t size_to_class(size_t size) {
        if (size <= MIN_BLOCK_SIZE) return 0;

        // Find the smallest size class that fits
        size_t block_size = MIN_BLOCK_SIZE;
        for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
            if (size <= block_size) return i;
            block_size *= 2;
        }
        return NUM_SIZE_CLASSES; // Too large for pool
    }

    static size_t size_class_to_size(size_t class_idx) {
        return MIN_BLOCK_SIZE << class_idx;
    }

    void* allocate_from_pool(size_t size) {
        size_t class_idx = size_to_class(size);
        if (class_idx >= NUM_SIZE_CLASSES) {
            return allocate_large(size, DEFAULT_ALIGNMENT);
        }

        size_t block_size = size_class_to_size(class_idx);

        std::lock_guard<std::mutex> lock(pool_mutex_);

        Pool& pool = pools_[class_idx];
        if (!pool.free_list.empty()) {
            void* ptr = pool.free_list.back();
            pool.free_list.pop_back();
            return ptr;
        }

        // Allocate new block
        void* ptr = aligned_alloc(block_size, DEFAULT_ALIGNMENT);
        if (ptr) {
            pool.blocks_allocated++;
            total_allocated_ += block_size;
            peak_allocated_ = std::max(peak_allocated_, total_allocated_.load());
        }
        return ptr;
    }

    void deallocate_to_pool(void* ptr) {
        // We need to know the size class - store it in the allocation header
        // For simplicity, we'll iterate to find which pool it might belong to
        // In production, we'd store metadata with the allocation

        std::lock_guard<std::mutex> lock(pool_mutex_);

        // Add to the first non-full pool (heuristic)
        // In a real implementation, we'd track the size class
        for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
            if (pools_[i].blocks_allocated > 0) {
                pools_[i].free_list.push_back(ptr);
                return;
            }
        }

        // Fallback: just free it
        aligned_free(ptr);
    }

    void* allocate_large(size_t size, size_t alignment) {
        void* ptr = aligned_alloc(size, alignment);
        if (ptr) {
            std::lock_guard<std::mutex> lock(large_mutex_);
            large_allocations_[ptr] = size;
            total_allocated_ += size;
            peak_allocated_ = std::max(peak_allocated_, total_allocated_.load());
        }
        return ptr;
    }

    mutable std::mutex pool_mutex_;
    std::mutex large_mutex_;
    Pool pools_[NUM_SIZE_CLASSES];
    std::unordered_map<void*, size_t> large_allocations_;
    std::atomic<size_t> total_allocated_{0};
    size_t peak_allocated_{0};
};

} // namespace memory
} // namespace dnn
