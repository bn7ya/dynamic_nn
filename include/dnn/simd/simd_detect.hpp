#pragma once

#include <cstdint>
#include <string>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace dnn {
namespace simd {

/**
 * SIMD instruction set levels, ordered by capability.
 */
enum class SIMDLevel {
    Scalar = 0,  // No SIMD
    SSE,         // 128-bit
    SSE2,
    SSE3,
    SSSE3,
    SSE4_1,
    SSE4_2,
    AVX,         // 256-bit
    AVX2,
    FMA,
    AVX512F,     // 512-bit
    AVX512DQ,
    AVX512VL
};

/**
 * Detects CPU features at runtime.
 * Thread-safe singleton.
 */
class CPUFeatures {
public:
    static const CPUFeatures& instance() {
        static CPUFeatures instance;
        return instance;
    }

    CPUFeatures(const CPUFeatures&) = delete;
    CPUFeatures& operator=(const CPUFeatures&) = delete;

    // SSE features
    bool has_sse() const { return sse_; }
    bool has_sse2() const { return sse2_; }
    bool has_sse3() const { return sse3_; }
    bool has_ssse3() const { return ssse3_; }
    bool has_sse4_1() const { return sse4_1_; }
    bool has_sse4_2() const { return sse4_2_; }

    // AVX features
    bool has_avx() const { return avx_; }
    bool has_avx2() const { return avx2_; }
    bool has_fma() const { return fma_; }

    // AVX-512 features
    bool has_avx512f() const { return avx512f_; }
    bool has_avx512dq() const { return avx512dq_; }
    bool has_avx512vl() const { return avx512vl_; }

    /**
     * Returns the best available SIMD level.
     */
    SIMDLevel best_available() const {
        if (avx512vl_) return SIMDLevel::AVX512VL;
        if (avx512dq_) return SIMDLevel::AVX512DQ;
        if (avx512f_) return SIMDLevel::AVX512F;
        if (fma_) return SIMDLevel::FMA;
        if (avx2_) return SIMDLevel::AVX2;
        if (avx_) return SIMDLevel::AVX;
        if (sse4_2_) return SIMDLevel::SSE4_2;
        if (sse4_1_) return SIMDLevel::SSE4_1;
        if (ssse3_) return SIMDLevel::SSSE3;
        if (sse3_) return SIMDLevel::SSE3;
        if (sse2_) return SIMDLevel::SSE2;
        if (sse_) return SIMDLevel::SSE;
        return SIMDLevel::Scalar;
    }

    /**
     * Returns a human-readable string for the SIMD level.
     */
    static std::string level_name(SIMDLevel level) {
        switch (level) {
            case SIMDLevel::Scalar: return "Scalar";
            case SIMDLevel::SSE: return "SSE";
            case SIMDLevel::SSE2: return "SSE2";
            case SIMDLevel::SSE3: return "SSE3";
            case SIMDLevel::SSSE3: return "SSSE3";
            case SIMDLevel::SSE4_1: return "SSE4.1";
            case SIMDLevel::SSE4_2: return "SSE4.2";
            case SIMDLevel::AVX: return "AVX";
            case SIMDLevel::AVX2: return "AVX2";
            case SIMDLevel::FMA: return "FMA";
            case SIMDLevel::AVX512F: return "AVX-512F";
            case SIMDLevel::AVX512DQ: return "AVX-512DQ";
            case SIMDLevel::AVX512VL: return "AVX-512VL";
            default: return "Unknown";
        }
    }

    /**
     * Returns the SIMD register width in bits.
     */
    static size_t register_width(SIMDLevel level) {
        switch (level) {
            case SIMDLevel::Scalar:
                return 64;
            case SIMDLevel::SSE:
            case SIMDLevel::SSE2:
            case SIMDLevel::SSE3:
            case SIMDLevel::SSSE3:
            case SIMDLevel::SSE4_1:
            case SIMDLevel::SSE4_2:
                return 128;
            case SIMDLevel::AVX:
            case SIMDLevel::AVX2:
            case SIMDLevel::FMA:
                return 256;
            case SIMDLevel::AVX512F:
            case SIMDLevel::AVX512DQ:
            case SIMDLevel::AVX512VL:
                return 512;
            default:
                return 64;
        }
    }

    /**
     * Cache line size in bytes (typically 64).
     */
    size_t cache_line_size() const { return cache_line_size_; }

    /**
     * L1 data cache size in bytes.
     */
    size_t l1_cache_size() const { return l1_cache_size_; }

    /**
     * L2 cache size in bytes.
     */
    size_t l2_cache_size() const { return l2_cache_size_; }

    /**
     * Number of logical CPU cores.
     */
    size_t num_cores() const { return num_cores_; }

private:
    CPUFeatures() { detect(); }

    void detect() {
        uint32_t eax, ebx, ecx, edx;

        // Get highest function parameter
        cpuid(0, 0, eax, ebx, ecx, edx);
        uint32_t max_id = eax;

        if (max_id >= 1) {
            cpuid(1, 0, eax, ebx, ecx, edx);

            // EDX features
            sse_ = (edx >> 25) & 1;
            sse2_ = (edx >> 26) & 1;

            // ECX features
            sse3_ = (ecx >> 0) & 1;
            ssse3_ = (ecx >> 9) & 1;
            sse4_1_ = (ecx >> 19) & 1;
            sse4_2_ = (ecx >> 20) & 1;
            fma_ = (ecx >> 12) & 1;

            // Check for AVX (requires OS support)
            bool os_avx_support = false;
            if ((ecx >> 27) & 1) { // OSXSAVE
                // Check XCR0
                uint32_t xcr0_eax, xcr0_edx;
                xgetbv(0, xcr0_eax, xcr0_edx);
                os_avx_support = ((xcr0_eax & 0x6) == 0x6); // XMM and YMM state
            }

            avx_ = ((ecx >> 28) & 1) && os_avx_support;
        }

        if (max_id >= 7) {
            cpuid(7, 0, eax, ebx, ecx, edx);

            // EBX features
            avx2_ = ((ebx >> 5) & 1) && avx_;
            avx512f_ = ((ebx >> 16) & 1) && avx_;
            avx512dq_ = ((ebx >> 17) & 1) && avx512f_;
            avx512vl_ = ((ebx >> 31) & 1) && avx512f_;
        }

        // Detect cache sizes (simplified)
        detect_cache_info();
        detect_cores();
    }

    void detect_cache_info() {
        // Default values
        cache_line_size_ = 64;
        l1_cache_size_ = 32 * 1024;  // 32 KB
        l2_cache_size_ = 256 * 1024; // 256 KB

        uint32_t eax, ebx, ecx, edx;
        cpuid(0x80000000, 0, eax, ebx, ecx, edx);

        if (eax >= 0x80000006) {
            cpuid(0x80000006, 0, eax, ebx, ecx, edx);
            cache_line_size_ = ecx & 0xFF;
            l2_cache_size_ = ((ecx >> 16) & 0xFFFF) * 1024;
        }
    }

    void detect_cores() {
        uint32_t eax, ebx, ecx, edx;
        cpuid(1, 0, eax, ebx, ecx, edx);
        num_cores_ = (ebx >> 16) & 0xFF;
        if (num_cores_ == 0) num_cores_ = 1;
    }

    static void cpuid(uint32_t func, uint32_t subfunc,
                      uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx) {
#ifdef _MSC_VER
        int regs[4];
        __cpuidex(regs, static_cast<int>(func), static_cast<int>(subfunc));
        eax = regs[0];
        ebx = regs[1];
        ecx = regs[2];
        edx = regs[3];
#else
        __cpuid_count(func, subfunc, eax, ebx, ecx, edx);
#endif
    }

    static void xgetbv(uint32_t index, uint32_t& eax, uint32_t& edx) {
#ifdef _MSC_VER
        unsigned long long result = _xgetbv(index);
        eax = static_cast<uint32_t>(result);
        edx = static_cast<uint32_t>(result >> 32);
#else
        __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
#endif
    }

    // CPU features
    bool sse_ = false;
    bool sse2_ = false;
    bool sse3_ = false;
    bool ssse3_ = false;
    bool sse4_1_ = false;
    bool sse4_2_ = false;
    bool avx_ = false;
    bool avx2_ = false;
    bool fma_ = false;
    bool avx512f_ = false;
    bool avx512dq_ = false;
    bool avx512vl_ = false;

    // Cache info
    size_t cache_line_size_ = 64;
    size_t l1_cache_size_ = 32 * 1024;
    size_t l2_cache_size_ = 256 * 1024;
    size_t num_cores_ = 1;
};

} // namespace simd
} // namespace dnn
