// include/pvec/pvec_distance.h — SIMD-accelerated vector distance metrics
// Copyright 2026 PomaiDB / pvec authors. MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include "pvec_platform.h"

namespace pvec {

// ── Scalar Implementations (Fallbacks & Verifiers) ───────────────────────────
namespace detail {

inline float dot_scalar(const float* a, const float* b, std::size_t n) noexcept {
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        sum0 += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        sum1 += static_cast<double>(a[i + 1]) * static_cast<double>(b[i + 1]);
        sum2 += static_cast<double>(a[i + 2]) * static_cast<double>(b[i + 2]);
        sum3 += static_cast<double>(a[i + 3]) * static_cast<double>(b[i + 3]);
    }
    double total = (sum0 + sum1) + (sum2 + sum3);
    for (; i < n; ++i) {
        total += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(total);
}

inline float l2_sq_scalar(const float* a, const float* b, std::size_t n) noexcept {
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        double d0 = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        double d1 = static_cast<double>(a[i + 1]) - static_cast<double>(b[i + 1]);
        double d2 = static_cast<double>(a[i + 2]) - static_cast<double>(b[i + 2]);
        double d3 = static_cast<double>(a[i + 3]) - static_cast<double>(b[i + 3]);
        sum0 += d0 * d0; sum1 += d1 * d1; sum2 += d2 * d2; sum3 += d3 * d3;
    }
    double total = (sum0 + sum1) + (sum2 + sum3);
    for (; i < n; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        total += d * d;
    }
    return static_cast<float>(total);
}

inline float manhattan_scalar(const float* a, const float* b, std::size_t n) noexcept {
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        total += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return static_cast<float>(total);
}

#if defined(PVEC_ARCH_X86_64) && (defined(__GNUC__) || defined(__clang__))
PVEC_TARGET_AVX2
inline float dot_avx2(const float* a, const float* b, std::size_t n) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }
    __m256 total = _mm256_add_ps(acc0, acc1);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    __m128 lo = _mm256_castps256_ps128(total);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

PVEC_TARGET_AVX2
inline float l2_sq_avx2(const float* a, const float* b, std::size_t n) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 d0 = _mm256_sub_ps(va0, vb0);
        __m256 d1 = _mm256_sub_ps(va1, vb1);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    for (; i + 7 < n; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }
    __m256 total = _mm256_add_ps(acc0, acc1);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    __m128 lo = _mm256_castps256_ps128(total);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
#endif

#if defined(PVEC_ARCH_ARM64)
inline float dot_neon(const float* a, const float* b, std::size_t n) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 7 < n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        acc0 = vfmaq_f32(acc0, va0, vb0);
        acc1 = vfmaq_f32(acc1, va1, vb1);
    }
    for (; i + 3 < n; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

inline float l2_sq_neon(const float* a, const float* b, std::size_t n) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 7 < n; i += 8) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = vfmaq_f32(acc0, d, d);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
#endif

} // namespace detail

// ── Public API ───────────────────────────────────────────────────────────────

/// Inner product: dot(a, b) = sum(a[i] * b[i])
inline float dot(std::span<const float> a, std::span<const float> b) noexcept {
    const std::size_t n = a.size();
    if (n == 0 || b.size() != n) return 0.0f;
#if defined(PVEC_ARCH_X86_64) && (defined(__GNUC__) || defined(__clang__))
    if (CpuFeatures::Get().has_avx2) {
        return detail::dot_avx2(a.data(), b.data(), n);
    }
#elif defined(PVEC_ARCH_ARM64)
    return detail::dot_neon(a.data(), b.data(), n);
#endif
    return detail::dot_scalar(a.data(), b.data(), n);
}

/// Squared Euclidean distance: sum((a[i] - b[i])^2)
inline float l2_sq(std::span<const float> a, std::span<const float> b) noexcept {
    const std::size_t n = a.size();
    if (n == 0 || b.size() != n) return 0.0f;
#if defined(PVEC_ARCH_X86_64) && (defined(__GNUC__) || defined(__clang__))
    if (CpuFeatures::Get().has_avx2) {
        return detail::l2_sq_avx2(a.data(), b.data(), n);
    }
#elif defined(PVEC_ARCH_ARM64)
    return detail::l2_sq_neon(a.data(), b.data(), n);
#endif
    return detail::l2_sq_scalar(a.data(), b.data(), n);
}

/// Euclidean distance: sqrt(sum((a[i] - b[i])^2))
inline float l2(std::span<const float> a, std::span<const float> b) noexcept {
    return std::sqrt(std::max(0.0f, l2_sq(a, b)));
}

/// Manhattan / L1 distance: sum(|a[i] - b[i]|)
inline float manhattan(std::span<const float> a, std::span<const float> b) noexcept {
    const std::size_t n = a.size();
    if (n == 0 || b.size() != n) return 0.0f;
    return detail::manhattan_scalar(a.data(), b.data(), n);
}

/// Cosine similarity: dot(a, b) / (||a|| * ||b||) in [-1.0, 1.0]
inline float cosine_similarity(std::span<const float> a, std::span<const float> b) noexcept {
    const std::size_t n = a.size();
    if (n == 0 || b.size() != n) return 0.0f;
    double dot_val = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double da = static_cast<double>(a[i]);
        double db = static_cast<double>(b[i]);
        dot_val += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }
    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom <= 1e-12) return 0.0f;
    float sim = static_cast<float>(dot_val / denom);
    return std::clamp(sim, -1.0f, 1.0f);
}

/// Cosine distance: 1.0 - cosine_similarity in [0.0, 2.0]
inline float cosine_distance(std::span<const float> a, std::span<const float> b) noexcept {
    return 1.0f - cosine_similarity(a, b);
}

/// Batch Dot Product against an n x dim row-major matrix
inline void dot_batch(std::span<const float> query,
                      const float* matrix,
                      std::size_t n,
                      std::size_t dim,
                      float* out) noexcept {
    if (!matrix || !out || dim == 0 || query.size() != dim) return;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = dot(query, std::span<const float>(matrix + i * dim, dim));
    }
}

/// Batch Squared L2 Distance against an n x dim row-major matrix
inline void l2_sq_batch(std::span<const float> query,
                        const float* matrix,
                        std::size_t n,
                        std::size_t dim,
                        float* out) noexcept {
    if (!matrix || !out || dim == 0 || query.size() != dim) return;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = l2_sq(query, std::span<const float>(matrix + i * dim, dim));
    }
}

} // namespace pvec
