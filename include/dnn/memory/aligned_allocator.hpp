#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <memory>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace dnn {
namespace memory {

// Default alignment for SIMD operations (AVX requires 32-byte alignment)
constexpr size_t DEFAULT_ALIGNMENT = 32;

/**
 * Allocates memory with specified alignment.
 * @param size Number of bytes to allocate
 * @param alignment Alignment requirement (must be power of 2)
 * @return Pointer to aligned memory, or nullptr on failure
 */
inline void* aligned_alloc(size_t size, size_t alignment = DEFAULT_ALIGNMENT) {
    if (size == 0) return nullptr;

    // Ensure alignment is at least sizeof(void*) and is a power of 2
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }

#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

/**
 * Frees memory allocated with aligned_alloc.
 */
inline void aligned_free(void* ptr) {
    if (ptr == nullptr) return;

#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * Custom deleter for aligned memory.
 */
template<typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const {
        if (ptr) {
            ptr->~T();
            aligned_free(ptr);
        }
    }
};

/**
 * Custom deleter for aligned arrays.
 */
template<typename T>
struct AlignedArrayDeleter {
    size_t count;

    explicit AlignedArrayDeleter(size_t n = 0) : count(n) {}

    void operator()(T* ptr) const {
        if (ptr) {
            // Call destructors for non-trivial types
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_t i = 0; i < count; ++i) {
                    ptr[i].~T();
                }
            }
            aligned_free(ptr);
        }
    }
};

/**
 * STL-compatible allocator for aligned memory.
 */
template<typename T, size_t Alignment = DEFAULT_ALIGNMENT>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    static constexpr size_t alignment = Alignment;

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(size_t n) {
        if (n == 0) return nullptr;

        void* ptr = aligned_alloc(n * sizeof(T), Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t /*n*/) noexcept {
        aligned_free(ptr);
    }

    template<typename U, typename... Args>
    void construct(U* ptr, Args&&... args) {
        new (ptr) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* ptr) {
        ptr->~U();
    }

    size_type max_size() const noexcept {
        return (static_cast<size_type>(-1) / sizeof(T));
    }

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

/**
 * Creates a unique_ptr with aligned memory.
 */
template<typename T, typename... Args>
std::unique_ptr<T, AlignedDeleter<T>> make_aligned_unique(Args&&... args) {
    void* raw = aligned_alloc(sizeof(T), alignof(T) > DEFAULT_ALIGNMENT ? alignof(T) : DEFAULT_ALIGNMENT);
    if (!raw) {
        throw std::bad_alloc();
    }
    try {
        T* ptr = new (raw) T(std::forward<Args>(args)...);
        return std::unique_ptr<T, AlignedDeleter<T>>(ptr);
    } catch (...) {
        aligned_free(raw);
        throw;
    }
}

/**
 * Creates an aligned array with unique_ptr ownership.
 */
template<typename T>
std::unique_ptr<T[], AlignedArrayDeleter<T>> make_aligned_array(size_t count) {
    if (count == 0) {
        return std::unique_ptr<T[], AlignedArrayDeleter<T>>(nullptr, AlignedArrayDeleter<T>(0));
    }

    void* raw = aligned_alloc(count * sizeof(T), DEFAULT_ALIGNMENT);
    if (!raw) {
        throw std::bad_alloc();
    }

    T* ptr = static_cast<T*>(raw);

    // Default construct elements for non-trivial types
    if constexpr (!std::is_trivially_default_constructible_v<T>) {
        try {
            for (size_t i = 0; i < count; ++i) {
                new (ptr + i) T();
            }
        } catch (...) {
            aligned_free(raw);
            throw;
        }
    }

    return std::unique_ptr<T[], AlignedArrayDeleter<T>>(ptr, AlignedArrayDeleter<T>(count));
}

} // namespace memory
} // namespace dnn
