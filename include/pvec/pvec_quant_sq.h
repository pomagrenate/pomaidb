// include/pvec/pvec_quant_sq.h — SIMD Scalar Quantization (SQ8 & SQ4)
// Copyright 2026 PomaiDB / pvec authors. MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
#include "pvec_platform.h"

namespace pvec {

namespace detail {

inline float dot_sq8_scalar(const float* q, const uint8_t* c, std::size_t n,
                            float min_val, float inv_scale, float q_sum) noexcept {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(q[i]) * static_cast<double>(c[i]);
    }
    return static_cast<float>(sum * static_cast<double>(inv_scale) + static_cast<double>(q_sum * min_val));
}

inline float l2_sq_sq8_scalar(const float* q, const uint8_t* c, std::size_t n,
                              float min_val, float max_val) noexcept {
    const float inv_scale = (max_val - min_val <= 1e-9f) ? 0.0f : ((max_val - min_val) / 255.0f);
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double val = static_cast<double>(min_val) + static_cast<double>(c[i]) * static_cast<double>(inv_scale);
        double diff = static_cast<double>(q[i]) - val;
        sum_sq += diff * diff;
    }
    return static_cast<float>(sum_sq);
}

#if defined(PVEC_ARCH_X86_64) && (defined(__GNUC__) || defined(__clang__))
PVEC_TARGET_AVX2
inline float dot_sq8_avx2(const float* q, const uint8_t* c, std::size_t n,
                          float min_val, float inv_scale, float q_sum) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c + i));
        __m256i i32_lo = _mm256_cvtepu8_epi32(raw16);
        __m128i raw_hi = _mm_srli_si128(raw16, 8);
        __m256i i32_hi = _mm256_cvtepu8_epi32(raw_hi);

        __m256 f32_lo = _mm256_cvtepi32_ps(i32_lo);
        __m256 f32_hi = _mm256_cvtepi32_ps(i32_hi);

        __m256 q_lo = _mm256_loadu_ps(q + i);
        __m256 q_hi = _mm256_loadu_ps(q + i + 8);

        acc0 = _mm256_fmadd_ps(q_lo, f32_lo, acc0);
        acc1 = _mm256_fmadd_ps(q_hi, f32_hi, acc1);
    }
    for (; i + 7 < n; i += 8) {
        __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c + i));
        __m256 f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw8));
        __m256 q8 = _mm256_loadu_ps(q + i);
        acc0 = _mm256_fmadd_ps(q8, f32, acc0);
    }
    __m256 total = _mm256_add_ps(acc0, acc1);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    __m128 lo = _mm256_castps256_ps128(total);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    for (; i < n; ++i) {
        sum += q[i] * static_cast<float>(c[i]);
    }
    return sum * inv_scale + q_sum * min_val;
}

PVEC_TARGET_AVX2
inline float l2_sq_sq8_avx2(const float* q, const uint8_t* c, std::size_t n,
                            float min_val, float max_val) noexcept {
    const float inv_scale = (max_val - min_val <= 1e-9f) ? 0.0f : ((max_val - min_val) / 255.0f);
    __m256 v_min = _mm256_set1_ps(min_val);
    __m256 v_scale = _mm256_set1_ps(inv_scale);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c + i));
        __m256i i32_lo = _mm256_cvtepu8_epi32(raw16);
        __m128i raw_hi = _mm_srli_si128(raw16, 8);
        __m256i i32_hi = _mm256_cvtepu8_epi32(raw_hi);

        __m256 deq_lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_lo), v_scale, v_min);
        __m256 deq_hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32_hi), v_scale, v_min);

        __m256 diff_lo = _mm256_sub_ps(_mm256_loadu_ps(q + i), deq_lo);
        __m256 diff_hi = _mm256_sub_ps(_mm256_loadu_ps(q + i + 8), deq_hi);

        acc0 = _mm256_fmadd_ps(diff_lo, diff_lo, acc0);
        acc1 = _mm256_fmadd_ps(diff_hi, diff_hi, acc1);
    }
    for (; i + 7 < n; i += 8) {
        __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c + i));
        __m256 deq = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw8)), v_scale, v_min);
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(q + i), deq);
        acc0 = _mm256_fmadd_ps(diff, diff, acc0);
    }
    __m256 total = _mm256_add_ps(acc0, acc1);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    __m128 lo = _mm256_castps256_ps128(total);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum_sq = _mm_cvtss_f32(sum128);

    for (; i < n; ++i) {
        float val = min_val + static_cast<float>(c[i]) * inv_scale;
        float diff = q[i] - val;
        sum_sq += diff * diff;
    }
    return sum_sq;
}
#endif

} // namespace detail

// ── SQ8 Quantizer ────────────────────────────────────────────────────────────

struct SQ8Params {
    float min_val{0.0f};
    float max_val{0.0f};
    float scale{1.0f};
    float inv_scale{1.0f};
};

class ScalarQuantizer8 {
public:
    static SQ8Params Calibrate(std::span<const float> vec) noexcept {
        if (vec.empty()) return {};
        float min_v = vec[0], max_v = vec[0];
        for (float v : vec) {
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        SQ8Params p;
        p.min_val = min_v;
        p.max_val = max_v;
        float range = max_v - min_v;
        if (range <= 1e-9f) {
            p.scale = 0.0f;
            p.inv_scale = 0.0f;
        } else {
            p.scale = 255.0f / range;
            p.inv_scale = range / 255.0f;
        }
        return p;
    }

    static void Encode(std::span<const float> in, std::span<uint8_t> out, const SQ8Params& p) noexcept {
        const std::size_t n = std::min(in.size(), out.size());
        for (std::size_t i = 0; i < n; ++i) {
            float norm = (in[i] - p.min_val) * p.scale;
            out[i] = static_cast<uint8_t>(std::clamp(std::round(norm), 0.0f, 255.0f));
        }
    }

    static void Decode(std::span<const uint8_t> in, std::span<float> out, const SQ8Params& p) noexcept {
        const std::size_t n = std::min(in.size(), out.size());
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = p.min_val + static_cast<float>(in[i]) * p.inv_scale;
        }
    }

    /// Fused SIMD Dot Product between float query and SQ8 codes
    static float Dot(std::span<const float> query, std::span<const uint8_t> codes,
                     float min_val, float inv_scale, float query_sum = 0.0f) noexcept {
        const std::size_t n = std::min(query.size(), codes.size());
        if (n == 0) return 0.0f;
        if (query_sum == 0.0f) {
            for (std::size_t i = 0; i < n; ++i) query_sum += query[i];
        }
#if defined(PVEC_ARCH_X86_64) && (defined(__GNUC__) || defined(__clang__))
        if (CpuFeatures::Get().has_avx2) {
            return detail::dot_sq8_avx2(query.data(), codes.data(), n, min_val, inv_scale, query_sum);
        }
#endif
        return detail::dot_sq8_scalar(query.data(), codes.data(), n, min_val, inv_scale, query_sum);
    }

    /// Fused SIMD Squared L2 Distance between float query and SQ8 codes
    static float L2Sq(std::span<const float> query, std::span<const uint8_t> codes,
                      float min_val, float max_val) noexcept {
        const std::size_t n = std::min(query.size(), codes.size());
        if (n == 0) return 0.0f;
#if defined(PVEC_ARCH_X86_64) && (defined(__GNUC__) || defined(__clang__))
        if (CpuFeatures::Get().has_avx2) {
            return detail::l2_sq_sq8_avx2(query.data(), codes.data(), n, min_val, max_val);
        }
#endif
        return detail::l2_sq_sq8_scalar(query.data(), codes.data(), n, min_val, max_val);
    }
};

// ── SQ4 Quantizer (4-bit nibbles, 2 values per byte) ─────────────────────────

class ScalarQuantizer4 {
public:
    static void Encode(std::span<const float> in, std::span<uint8_t> out, float min_v, float max_v) noexcept {
        const std::size_t n = in.size();
        const std::size_t out_bytes = (n + 1) / 2;
        if (out.size() < out_bytes) return;
        float range = max_v - min_v;
        float scale = (range <= 1e-9f) ? 0.0f : (15.0f / range);

        std::fill(out.begin(), out.begin() + out_bytes, 0);
        for (std::size_t i = 0; i < n; ++i) {
            float norm = (in[i] - min_v) * scale;
            uint8_t q = static_cast<uint8_t>(std::clamp(std::round(norm), 0.0f, 15.0f));
            if (i % 2 == 0) {
                out[i / 2] |= (q & 0x0F);
            } else {
                out[i / 2] |= ((q & 0x0F) << 4);
            }
        }
    }

    static void Decode(std::span<const uint8_t> in, std::span<float> out, float min_v, float max_v) noexcept {
        const std::size_t n = out.size();
        float range = max_v - min_v;
        float inv_scale = (range <= 1e-9f) ? 0.0f : (range / 15.0f);
        for (std::size_t i = 0; i < n; ++i) {
            uint8_t byte = in[i / 2];
            uint8_t q = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            out[i] = min_v + static_cast<float>(q) * inv_scale;
        }
    }
};

} // namespace pvec
