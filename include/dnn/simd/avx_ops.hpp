#pragma once

#include "simd_ops.hpp"

#if defined(__AVX__) || defined(__AVX2__) || defined(DNN_ENABLE_AVX)
#include <immintrin.h>
#endif

namespace dnn {
namespace simd {

#if defined(__AVX__) || defined(__AVX2__) || defined(DNN_ENABLE_AVX)

/**
 * AVX/AVX2 optimized operations for float.
 */
template<>
class AVXOps : public SIMDOps<float> {
public:
    void add(const float* a, const float* b, float* c, size_t n) const override {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    void sub(const float* a, const float* b, float* c, size_t n) const override {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] - b[i];
        }
    }

    void mul(const float* a, const float* b, float* c, size_t n) const override {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] * b[i];
        }
    }

    void div(const float* a, const float* b, float* c, size_t n) const override {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] / b[i];
        }
    }

    void add_scalar(const float* a, float scalar, float* c, size_t n) const override {
        __m256 vs = _mm256_set1_ps(scalar);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vc = _mm256_add_ps(va, vs);
            _mm256_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] + scalar;
        }
    }

    void mul_scalar(const float* a, float scalar, float* c, size_t n) const override {
        __m256 vs = _mm256_set1_ps(scalar);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vc = _mm256_mul_ps(va, vs);
            _mm256_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] * scalar;
        }
    }

    float sum(const float* a, size_t n) const override {
        __m256 vsum = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            vsum = _mm256_add_ps(vsum, va);
        }

        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 sum128 = _mm_add_ps(hi, lo);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);

        for (; i < n; ++i) {
            result += a[i];
        }
        return result;
    }

    float max(const float* a, size_t n) const override {
        if (n == 0) return 0.0f;

        __m256 vmax = _mm256_set1_ps(a[0]);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            vmax = _mm256_max_ps(vmax, va);
        }

        // Horizontal max
        __m128 hi = _mm256_extractf128_ps(vmax, 1);
        __m128 lo = _mm256_castps256_ps128(vmax);
        __m128 max128 = _mm_max_ps(hi, lo);
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
        float result = _mm_cvtss_f32(max128);

        for (; i < n; ++i) {
            if (a[i] > result) result = a[i];
        }
        return result;
    }

    float min(const float* a, size_t n) const override {
        if (n == 0) return 0.0f;

        __m256 vmin = _mm256_set1_ps(a[0]);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            vmin = _mm256_min_ps(vmin, va);
        }

        // Horizontal min
        __m128 hi = _mm256_extractf128_ps(vmin, 1);
        __m128 lo = _mm256_castps256_ps128(vmin);
        __m128 min128 = _mm_min_ps(hi, lo);
        min128 = _mm_min_ps(min128, _mm_shuffle_ps(min128, min128, _MM_SHUFFLE(2, 3, 0, 1)));
        min128 = _mm_min_ps(min128, _mm_shuffle_ps(min128, min128, _MM_SHUFFLE(1, 0, 3, 2)));
        float result = _mm_cvtss_f32(min128);

        for (; i < n; ++i) {
            if (a[i] < result) result = a[i];
        }
        return result;
    }

    float dot(const float* a, const float* b, size_t n) const override {
        __m256 vsum = _mm256_setzero_ps();
        size_t i = 0;

#if defined(__FMA__) || defined(DNN_ENABLE_AVX2)
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            vsum = _mm256_fmadd_ps(va, vb, vsum);
        }
#else
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
        }
#endif

        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 sum128 = _mm_add_ps(hi, lo);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);

        for (; i < n; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    void matmul(const float* a, const float* b, float* c,
               size_t m, size_t k, size_t n,
               size_t lda, size_t ldb, size_t ldc) const override {
        // Blocked matrix multiplication optimized for AVX
        constexpr size_t BLOCK_M = 64;
        constexpr size_t BLOCK_N = 64;
        constexpr size_t BLOCK_K = 64;

        // Zero output
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; j += 8) {
                if (j + 8 <= n) {
                    _mm256_storeu_ps(c + i * ldc + j, _mm256_setzero_ps());
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

                    // Process block with AVX
                    for (size_t i = i0; i < i_end; ++i) {
                        for (size_t kk = k0; kk < k_end; ++kk) {
                            __m256 va = _mm256_set1_ps(a[i * lda + kk]);
                            size_t j = j0;
                            for (; j + 8 <= j_end; j += 8) {
                                __m256 vb = _mm256_loadu_ps(b + kk * ldb + j);
                                __m256 vc = _mm256_loadu_ps(c + i * ldc + j);
#if defined(__FMA__) || defined(DNN_ENABLE_AVX2)
                                vc = _mm256_fmadd_ps(va, vb, vc);
#else
                                vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
#endif
                                _mm256_storeu_ps(c + i * ldc + j, vc);
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
        __m256 zero = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vc = _mm256_max_ps(va, zero);
            _mm256_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] > 0.0f ? a[i] : 0.0f;
        }
    }

    void relu_backward(const float* a, const float* grad, float* c, size_t n) const override {
        __m256 zero = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vg = _mm256_loadu_ps(grad + i);
            __m256 mask = _mm256_cmp_ps(va, zero, _CMP_GT_OQ);
            __m256 vc = _mm256_and_ps(mask, vg);
            _mm256_storeu_ps(c + i, vc);
        }
        for (; i < n; ++i) {
            c[i] = a[i] > 0.0f ? grad[i] : 0.0f;
        }
    }

    void sigmoid(const float* a, float* c, size_t n) const override {
        // Sigmoid is complex - use scalar for accuracy
        for (size_t i = 0; i < n; ++i) {
            c[i] = 1.0f / (1.0f + std::exp(-a[i]));
        }
    }

    void tanh(const float* a, float* c, size_t n) const override {
        // Tanh is complex - use scalar for accuracy
        for (size_t i = 0; i < n; ++i) {
            c[i] = std::tanh(a[i]);
        }
    }

    void copy(const float* src, float* dst, size_t n) const override {
        std::memcpy(dst, src, n * sizeof(float));
    }

    void fill(float* dst, float value, size_t n) const override {
        __m256 vval = _mm256_set1_ps(value);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(dst + i, vval);
        }
        for (; i < n; ++i) {
            dst[i] = value;
        }
    }
};

#endif // AVX support

} // namespace simd
} // namespace dnn
