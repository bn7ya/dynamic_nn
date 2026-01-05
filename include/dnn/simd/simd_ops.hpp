#pragma once

#include "simd_detect.hpp"
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace dnn {
namespace simd {

/**
 * Abstract interface for SIMD operations.
 * Implementations are selected at runtime based on CPU capabilities.
 */
template<typename T>
class SIMDOps {
public:
    virtual ~SIMDOps() = default;

    // Element-wise operations
    virtual void add(const T* a, const T* b, T* c, size_t n) const = 0;
    virtual void sub(const T* a, const T* b, T* c, size_t n) const = 0;
    virtual void mul(const T* a, const T* b, T* c, size_t n) const = 0;
    virtual void div(const T* a, const T* b, T* c, size_t n) const = 0;

    // Scalar operations
    virtual void add_scalar(const T* a, T scalar, T* c, size_t n) const = 0;
    virtual void mul_scalar(const T* a, T scalar, T* c, size_t n) const = 0;

    // Reductions
    virtual T sum(const T* a, size_t n) const = 0;
    virtual T max(const T* a, size_t n) const = 0;
    virtual T min(const T* a, size_t n) const = 0;
    virtual T dot(const T* a, const T* b, size_t n) const = 0;

    // Matrix operations
    virtual void matmul(const T* a, const T* b, T* c,
                       size_t m, size_t k, size_t n,
                       size_t lda, size_t ldb, size_t ldc) const = 0;

    // Activation functions
    virtual void relu(const T* a, T* c, size_t n) const = 0;
    virtual void relu_backward(const T* a, const T* grad, T* c, size_t n) const = 0;
    virtual void sigmoid(const T* a, T* c, size_t n) const = 0;
    virtual void tanh(const T* a, T* c, size_t n) const = 0;

    // Utility
    virtual void copy(const T* src, T* dst, size_t n) const = 0;
    virtual void fill(T* dst, T value, size_t n) const = 0;

    // Factory method - creates best implementation for current CPU
    static std::unique_ptr<SIMDOps<T>> create();
    static std::unique_ptr<SIMDOps<T>> create(SIMDLevel level);
};

/**
 * Scalar (fallback) implementation.
 */
template<typename T>
class ScalarOps : public SIMDOps<T> {
public:
    void add(const T* a, const T* b, T* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    void sub(const T* a, const T* b, T* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] - b[i];
        }
    }

    void mul(const T* a, const T* b, T* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] * b[i];
        }
    }

    void div(const T* a, const T* b, T* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] / b[i];
        }
    }

    void add_scalar(const T* a, T scalar, T* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] + scalar;
        }
    }

    void mul_scalar(const T* a, T scalar, T* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] * scalar;
        }
    }

    T sum(const T* a, size_t n) const override {
        T result = T(0);
        for (size_t i = 0; i < n; ++i) {
            result += a[i];
        }
        return result;
    }

    T max(const T* a, size_t n) const override {
        if (n == 0) return T(0);
        T result = a[0];
        for (size_t i = 1; i < n; ++i) {
            if (a[i] > result) result = a[i];
        }
        return result;
    }

    T min(const T* a, size_t n) const override {
        if (n == 0) return T(0);
        T result = a[0];
        for (size_t i = 1; i < n; ++i) {
            if (a[i] < result) result = a[i];
        }
        return result;
    }

    T dot(const T* a, const T* b, size_t n) const override {
        T result = T(0);
        for (size_t i = 0; i < n; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    void matmul(const T* a, const T* b, T* c,
               size_t m, size_t k, size_t n,
               size_t lda, size_t ldb, size_t ldc) const override {
        // Simple cache-friendly blocked matrix multiplication
        constexpr size_t BLOCK = 32;

        // Zero output
        for (size_t i = 0; i < m; ++i) {
            std::memset(c + i * ldc, 0, n * sizeof(T));
        }

        for (size_t i0 = 0; i0 < m; i0 += BLOCK) {
            size_t i_end = std::min(i0 + BLOCK, m);
            for (size_t k0 = 0; k0 < k; k0 += BLOCK) {
                size_t k_end = std::min(k0 + BLOCK, k);
                for (size_t j0 = 0; j0 < n; j0 += BLOCK) {
                    size_t j_end = std::min(j0 + BLOCK, n);

                    // Process block
                    for (size_t i = i0; i < i_end; ++i) {
                        for (size_t kk = k0; kk < k_end; ++kk) {
                            T a_ik = a[i * lda + kk];
                            for (size_t j = j0; j < j_end; ++j) {
                                c[i * ldc + j] += a_ik * b[kk * ldb + j];
                            }
                        }
                    }
                }
            }
        }
    }

    void relu(const T* a, T* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] > T(0) ? a[i] : T(0);
        }
    }

    void relu_backward(const T* a, const T* grad, T* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] > T(0) ? grad[i] : T(0);
        }
    }

    void sigmoid(const T* a, T* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = T(1) / (T(1) + std::exp(-a[i]));
        }
    }

    void tanh(const T* a, T* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = std::tanh(a[i]);
        }
    }

    void copy(const T* src, T* dst, size_t n) const override {
        std::memcpy(dst, src, n * sizeof(T));
    }

    void fill(T* dst, T value, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = value;
        }
    }
};

} // namespace simd
} // namespace dnn
