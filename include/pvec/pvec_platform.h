// include/pvec/pvec_platform.h — Platform detection, alignment, and CPU feature dispatch
// Copyright 2026 PomaiDB / pvec authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

// ── Architecture Detection ───────────────────────────────────────────────────
#if defined(__x86_64__) || defined(_M_X64)
    #define PVEC_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define PVEC_ARCH_ARM64 1
#elif defined(__riscv) && (__riscv_xlen == 64)
    #define PVEC_ARCH_RISCV64 1
#endif

// ── Compiler Detection ───────────────────────────────────────────────────────
#if defined(__GNUC__) || defined(__clang__)
    #define PVEC_ALWAYS_INLINE inline __attribute__((always_inline))
    #define PVEC_TARGET_AVX2 __attribute__((target("avx2,fma")))
    #define PVEC_TARGET_AVX512 __attribute__((target("avx512f,avx512dq,avx512vl,fma")))
#elif defined(_MSC_VER)
    #define PVEC_ALWAYS_INLINE __forceinline
    #define PVEC_TARGET_AVX2
    #define PVEC_TARGET_AVX512
#else
    #define PVEC_ALWAYS_INLINE inline
    #define PVEC_TARGET_AVX2
    #define PVEC_TARGET_AVX512
#endif

// ── SIMD Headers ─────────────────────────────────────────────────────────────
#if defined(PVEC_ARCH_X86_64)
    #include <immintrin.h>
#elif defined(PVEC_ARCH_ARM64)
    #include <arm_neon.h>
    #if defined(__ARM_FEATURE_SVE)
        #include <arm_sve.h>
    #endif
#endif

namespace pvec {

// Standard cache line alignment for zero false sharing and SIMD efficiency
constexpr std::size_t kAlign = 64;

// ── Aligned Allocation ───────────────────────────────────────────────────────
inline void* aligned_alloc(std::size_t bytes, std::size_t alignment = kAlign) noexcept {
    if (bytes == 0) return nullptr;
    alignment = (alignment < sizeof(void*)) ? sizeof(void*) : alignment;
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#elif defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    return ::aligned_alloc(alignment, (bytes + alignment - 1) & ~(alignment - 1));
#else
    void* ptr = nullptr;
    if (::posix_memalign(&ptr, alignment, bytes) != 0) return nullptr;
    return ptr;
#endif
}

inline void aligned_free(void* ptr) noexcept {
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

// ── CPU Feature Detection ───────────────────────────────────────────────────
struct CpuFeatures {
    bool has_avx2{false};
    bool has_avx512{false};
    bool has_f16c{false};
    bool has_neon{false};
    bool has_sve{false};

    static const CpuFeatures& Get() noexcept {
        static CpuFeatures features = Detect();
        return features;
    }

private:
    static CpuFeatures Detect() noexcept {
        CpuFeatures f;
#if defined(PVEC_ARCH_X86_64) && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        f.has_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
        f.has_avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl");
        f.has_f16c = __builtin_cpu_supports("f16c");
#elif defined(PVEC_ARCH_X86_64) && defined(_MSC_VER)
        f.has_avx2 = true; // Assumed supported on modern x86_64 targets
        f.has_f16c = true;
#elif defined(PVEC_ARCH_ARM64)
        f.has_neon = true; // Mandatory on AArch64
#endif
        return f;
    }
};

} // namespace pvec
