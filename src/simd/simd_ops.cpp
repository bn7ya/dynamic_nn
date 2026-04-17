// simd_ops.cpp - Factory method implementations for SIMDOps

#include "dnn/simd/simd_ops.hpp"

// Include AVX implementations when available
#if defined(__AVX__) || defined(__AVX2__) || defined(DNN_ENABLE_AVX) || \
    (defined(_MSC_VER) && defined(__AVX2__))
#include "dnn/simd/avx_ops.hpp"
#endif

namespace dnn {
namespace simd {

// Explicit template instantiation for factory methods

template<>
std::unique_ptr<SIMDOps<float>> SIMDOps<float>::create(SIMDLevel level) {
#if defined(__AVX__) || defined(__AVX2__) || defined(DNN_ENABLE_AVX) || \
    (defined(_MSC_VER) && defined(__AVX2__))
    switch (level) {
        case SIMDLevel::AVX512F:
        case SIMDLevel::AVX512DQ:
        case SIMDLevel::AVX512VL:
            // Fall through to AVX2 (no AVX-512 implementation yet)
        case SIMDLevel::FMA:
        case SIMDLevel::AVX2:
        case SIMDLevel::AVX:
            return std::make_unique<AVXOps<float>>();
        case SIMDLevel::SSE4_2:
        case SIMDLevel::SSE4_1:
        case SIMDLevel::SSSE3:
        case SIMDLevel::SSE3:
        case SIMDLevel::SSE2:
        case SIMDLevel::SSE:
            // Fall through to scalar (no SSE implementation yet)
        case SIMDLevel::Scalar:
        default:
            return std::make_unique<ScalarOps<float>>();
    }
#else
    (void)level;
    return std::make_unique<ScalarOps<float>>();
#endif
}

template<>
std::unique_ptr<SIMDOps<float>> SIMDOps<float>::create() {
    return create(CPUFeatures::instance().best_available());
}

template<>
std::unique_ptr<SIMDOps<double>> SIMDOps<double>::create(SIMDLevel level) {
    // Double precision AVX uses 4-wide vectors (vs 8-wide for float)
    // For now, use scalar implementation for double
    // TODO: Add AVX double implementation
    (void)level;
    return std::make_unique<ScalarOps<double>>();
}

template<>
std::unique_ptr<SIMDOps<double>> SIMDOps<double>::create() {
    return create(CPUFeatures::instance().best_available());
}

} // namespace simd
} // namespace dnn
