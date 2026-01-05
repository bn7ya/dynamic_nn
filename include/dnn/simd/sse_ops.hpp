#pragma once

#include "simd_ops.hpp"

#if defined(__SSE__) || defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#include <emmintrin.h>
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#endif

namespace dnn {
namespace simd {

#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)

/**
 * SSE2 optimized operations for float.
 */
template<>
class SSEOps : public SIMDOps<float> {
public:
    void add(const float* a, const float* b, float* c, size_t n) const override {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vc = _mm_add_ps(va, vb);
            _mm_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    void sub(const float* a, const float* b, float* c, size_t n) const override {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vc = _mm_sub_ps(va, vb);
            _mm_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] - b[i];
        }
    }

    void mul(const float* a, const float* b, float* c, size_t n) const override {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vc = _mm_mul_ps(va, vb);
            _mm_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] * b[i];
        }
    }

    void div(const float* a, const float* b, float* c, size_t n) const override {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vc = _mm_div_ps(va, vb);
            _mm_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] / b[i];
        }
    }

    void add_scalar(const float* a, float scalar, float* c, size_t n) const override {
        __m128 vs = _mm_set1_ps(scalar);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vc = _mm_add_ps(va, vs);
            _mm_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] + scalar;
        }
    }

    void mul_scalar(const float* a, float scalar, float* c, size_t n) const override {
        __m128 vs = _mm_set1_ps(scalar);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vc = _mm_mul_ps(va, vs);
            _mm_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] * scalar;
        }
    }

    float sum(const float* a, size_t n) const override {
        __m128 vsum = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            vsum = _mm_add_ps(vsum, va);
        }

        // Horizontal sum
#ifdef __SSE3__
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
#else
        __m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
        vsum = _mm_add_ps(vsum, shuf);
        shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(1, 0, 3, 2));
        vsum = _mm_add_ps(vsum, shuf);
#endif
        float result = _mm_cvtss_f32(vsum);

        for (; i < n; ++i) {
            result += a[i];
        }
        return result;
    }

    float max(const float* a, size_t n) const override {
        if (n == 0) return 0.0f;

        __m128 vmax = _mm_set1_ps(a[0]);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            vmax = _mm_max_ps(vmax, va);
        }

        // Horizontal max
        __m128 shuf = _mm_shuffle_ps(vmax, vmax, _MM_SHUFFLE(2, 3, 0, 1));
        vmax = _mm_max_ps(vmax, shuf);
        shuf = _mm_shuffle_ps(vmax, vmax, _MM_SHUFFLE(1, 0, 3, 2));
        vmax = _mm_max_ps(vmax, shuf);
        float result = _mm_cvtss_f32(vmax);

        for (; i < n; ++i) {
            if (a[i] > result) result = a[i];
        }
        return result;
    }

    float min(const float* a, size_t n) const override {
        if (n == 0) return 0.0f;

        __m128 vmin = _mm_set1_ps(a[0]);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            vmin = _mm_min_ps(vmin, va);
        }

        // Horizontal min
        __m128 shuf = _mm_shuffle_ps(vmin, vmin, _MM_SHUFFLE(2, 3, 0, 1));
        vmin = _mm_min_ps(vmin, shuf);
        shuf = _mm_shuffle_ps(vmin, vmin, _MM_SHUFFLE(1, 0, 3, 2));
        vmin = _mm_min_ps(vmin, shuf);
        float result = _mm_cvtss_f32(vmin);

        for (; i < n; ++i) {
            if (a[i] < result) result = a[i];
        }
        return result;
    }

    float dot(const float* a, const float* b, size_t n) const override {
        __m128 vsum = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            vsum = _mm_add_ps(vsum, _mm_mul_ps(va, vb));
        }

        // Horizontal sum
#ifdef __SSE3__
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
#else
        __m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
        vsum = _mm_add_ps(vsum, shuf);
        shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(1, 0, 3, 2));
        vsum = _mm_add_ps(vsum, shuf);
#endif
        float result = _mm_cvtss_f32(vsum);

        for (; i < n; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    void matmul(const float* a, const float* b, float* c,
               size_t m, size_t k, size_t n,
               size_t lda, size_t ldb, size_t ldc) const override {
        // Blocked matrix multiplication optimized for SSE
        constexpr size_t BLOCK_M = 32;
        constexpr size_t BLOCK_N = 32;
        constexpr size_t BLOCK_K = 32;

        // Zero output
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; j += 4) {
                if (j + 4 <= n) {
                    _mm_storeu_ps(c + i * ldc + j, _mm_setzero_ps());
                } else {
                    for (size_t jj = j; jj < n; ++jj) {
                        c[i * ldc + jj] = 0.0f;
                    }
                }
            }
        }

        for (size_t i0 = 0; i0 < m; i0 += BLOCK_M) {
            size_t i_end = std::min(i0 + BLOCK_M, m);
            for (size_t k0 = 0; k0 < k; k0 += BLOCK_K) {
                size_t k_end = std::min(k0 + BLOCK_K, k);
                for (size_t j0 = 0; j0 < n; j0 += BLOCK_N) {
                    size_t j_end = std::min(j0 + BLOCK_N, n);

                    // Process block with SSE
                    for (size_t i = i0; i < i_end; ++i) {
                        for (size_t kk = k0; kk < k_end; ++kk) {
                            __m128 va = _mm_set1_ps(a[i * lda + kk]);
                            size_t j = j0;
                            for (; j + 4 <= j_end; j += 4) {
                                __m128 vb = _mm_loadu_ps(b + kk * ldb + j);
                                __m128 vc = _mm_loadu_ps(c + i * ldc + j);
                                vc = _mm_add_ps(vc, _mm_mul_ps(va, vb));
                                _mm_storeu_ps(c + i * ldc + j, vc);
                            }
                            // Handle remainder
                            for (; j < j_end; ++j) {
                                c[i * ldc + j] += a[i * lda + kk] * b[kk * ldb + j];
                            }
                        }
                    }
                }
            }
        }
    }

    void relu(const float* a, float* c, size_t n) const override {
        __m128 zero = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vc = _mm_max_ps(va, zero);
            _mm_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] > 0.0f ? a[i] : 0.0f;
        }
    }

    void relu_backward(const float* a, const float* grad, float* c, size_t n) const override {
        __m128 zero = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vg = _mm_loadu_ps(grad + i);
            __m128 mask = _mm_cmpgt_ps(va, zero);
            __m128 vc = _mm_and_ps(mask, vg);
            _mm_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] > 0.0f ? grad[i] : 0.0f;
        }
    }

    void sigmoid(const float* a, float* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = 1.0f / (1.0f + std::exp(-a[i]));
        }
    }

    void tanh(const float* a, float* c, size_t n) const override {
        for (size_t i = 0; i < n; ++i) {
            c[i] = std::tanh(a[i]);
        }
    }

    void copy(const float* src, float* dst, size_t n) const override {
        std::memcpy(dst, src, n * sizeof(float));
    }

    void fill(float* dst, float value, size_t n) const override {
        __m128 vval = _mm_set1_ps(value);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm_storeu_ps(dst + i, vval);
        }
        for (; i < n; ++i) {
            dst[i] = value;
        }
    }
};

#endif // SSE2 support

} // namespace simd
} // namespace dnn
