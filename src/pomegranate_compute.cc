// src/pomegranate_compute.cc — Custom Core Compute Accelerator for PomaiDB
// Copyright 2026 PomaiDB authors. MIT License.

#include "pomegranate_compute.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace pomai::compute {

namespace {

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
#endif
inline float HorizontalSum(__m256 v) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
#else
    return 0.0f;
#endif
}

// Portable scalar fallback for DotSq8_4x
void DotSq8_4x_Scalar(const float* query,
                      const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3,
                      size_t dim, float min_val, float inv_scale, float query_sum,
                      float out_scores[4]) noexcept {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double q = static_cast<double>(query[i]);
        s0 += q * static_cast<double>(c0[i]);
        s1 += q * static_cast<double>(c1[i]);
        s2 += q * static_cast<double>(c2[i]);
        s3 += q * static_cast<double>(c3[i]);
    }
    double offset = static_cast<double>(query_sum * min_val);
    double scale = static_cast<double>(inv_scale);
    out_scores[0] = static_cast<float>(s0 * scale + offset);
    out_scores[1] = static_cast<float>(s1 * scale + offset);
    out_scores[2] = static_cast<float>(s2 * scale + offset);
    out_scores[3] = static_cast<float>(s3 * scale + offset);
}

// Portable scalar fallback for L2SqSq8_4x
void L2SqSq8_4x_Scalar(const float* query,
                       const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3,
                       size_t dim, float min_val, float max_val,
                       float out_dists[4]) noexcept {
    const float inv_scale = (max_val - min_val <= 1e-9f) ? 0.0f : ((max_val - min_val) / 255.0f);
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    double m = static_cast<double>(min_val);
    double scale = static_cast<double>(inv_scale);

    for (size_t i = 0; i < dim; ++i) {
        double q = static_cast<double>(query[i]);
        double d0 = q - (m + static_cast<double>(c0[i]) * scale);
        double d1 = q - (m + static_cast<double>(c1[i]) * scale);
        double d2 = q - (m + static_cast<double>(c2[i]) * scale);
        double d3 = q - (m + static_cast<double>(c3[i]) * scale);
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }
    out_dists[0] = static_cast<float>(sum0);
    out_dists[1] = static_cast<float>(sum1);
    out_dists[2] = static_cast<float>(sum2);
    out_dists[3] = static_cast<float>(sum3);
}

// Portable scalar fallback for DotF32_4x
void DotF32_4x_Scalar(const float* query,
                      const float* v0, const float* v1, const float* v2, const float* v3,
                      size_t dim, float out_dots[4]) noexcept {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double q = static_cast<double>(query[i]);
        s0 += q * static_cast<double>(v0[i]);
        s1 += q * static_cast<double>(v1[i]);
        s2 += q * static_cast<double>(v2[i]);
        s3 += q * static_cast<double>(v3[i]);
    }
    out_dots[0] = static_cast<float>(s0);
    out_dots[1] = static_cast<float>(s1);
    out_dots[2] = static_cast<float>(s2);
    out_dots[3] = static_cast<float>(s3);
}

// Portable scalar fallback for L2SqF32_4x
void L2SqF32_4x_Scalar(const float* query,
                       const float* v0, const float* v1, const float* v2, const float* v3,
                       size_t dim, float out_l2sq[4]) noexcept {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double q = static_cast<double>(query[i]);
        double d0 = q - static_cast<double>(v0[i]);
        double d1 = q - static_cast<double>(v1[i]);
        double d2 = q - static_cast<double>(v2[i]);
        double d3 = q - static_cast<double>(v3[i]);
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    out_l2sq[0] = static_cast<float>(s0);
    out_l2sq[1] = static_cast<float>(s1);
    out_l2sq[2] = static_cast<float>(s2);
    out_l2sq[3] = static_cast<float>(s3);
}

// Portable scalar fallback for SumF32
float SumF32_Scalar(const float* data, size_t dim) noexcept {
    double s = 0.0;
    for (size_t i = 0; i < dim; ++i) s += static_cast<double>(data[i]);
    return static_cast<float>(s);
}

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))

__attribute__((target("avx2,fma")))
void DotSq8_4x_Avx2(const float* query,
                    const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3,
                    size_t dim, float min_val, float inv_scale, float query_sum,
                    float out_scores[4]) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 q_lo = _mm256_loadu_ps(query + i);
        __m256 q_hi = _mm256_loadu_ps(query + i + 8);

        // Vector 0
        __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c0 + i));
        __m256 f0_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r0));
        __m256 f0_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(r0, 8)));
        acc0 = _mm256_fmadd_ps(q_lo, f0_lo, acc0);
        acc0 = _mm256_fmadd_ps(q_hi, f0_hi, acc0);

        // Vector 1
        __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c1 + i));
        __m256 f1_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r1));
        __m256 f1_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(r1, 8)));
        acc1 = _mm256_fmadd_ps(q_lo, f1_lo, acc1);
        acc1 = _mm256_fmadd_ps(q_hi, f1_hi, acc1);

        // Vector 2
        __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c2 + i));
        __m256 f2_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r2));
        __m256 f2_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(r2, 8)));
        acc2 = _mm256_fmadd_ps(q_lo, f2_lo, acc2);
        acc2 = _mm256_fmadd_ps(q_hi, f2_hi, acc2);

        // Vector 3
        __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c3 + i));
        __m256 f3_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r3));
        __m256 f3_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(r3, 8)));
        acc3 = _mm256_fmadd_ps(q_lo, f3_lo, acc3);
        acc3 = _mm256_fmadd_ps(q_hi, f3_hi, acc3);
    }

    for (; i + 7 < dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query + i);

        __m128i r0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0 + i));
        acc0 = _mm256_fmadd_ps(q, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r0)), acc0);

        __m128i r1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1 + i));
        acc1 = _mm256_fmadd_ps(q, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r1)), acc1);

        __m128i r2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2 + i));
        acc2 = _mm256_fmadd_ps(q, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r2)), acc2);

        __m128i r3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3 + i));
        acc3 = _mm256_fmadd_ps(q, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r3)), acc3);
    }

    float s0 = HorizontalSum(acc0);
    float s1 = HorizontalSum(acc1);
    float s2 = HorizontalSum(acc2);
    float s3 = HorizontalSum(acc3);

    for (; i < dim; ++i) {
        float q = query[i];
        s0 += q * static_cast<float>(c0[i]);
        s1 += q * static_cast<float>(c1[i]);
        s2 += q * static_cast<float>(c2[i]);
        s3 += q * static_cast<float>(c3[i]);
    }

    float offset = query_sum * min_val;
    out_scores[0] = s0 * inv_scale + offset;
    out_scores[1] = s1 * inv_scale + offset;
    out_scores[2] = s2 * inv_scale + offset;
    out_scores[3] = s3 * inv_scale + offset;
}

__attribute__((target("avx2,fma")))
void L2SqSq8_4x_Avx2(const float* query,
                     const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3,
                     size_t dim, float min_val, float max_val,
                     float out_dists[4]) noexcept {
    const float inv_scale = (max_val - min_val <= 1e-9f) ? 0.0f : ((max_val - min_val) / 255.0f);
    __m256 v_min = _mm256_set1_ps(min_val);
    __m256 v_scale = _mm256_set1_ps(inv_scale);

    __m256 acc0_a = _mm256_setzero_ps();
    __m256 acc0_b = _mm256_setzero_ps();
    __m256 acc1_a = _mm256_setzero_ps();
    __m256 acc1_b = _mm256_setzero_ps();
    __m256 acc2_a = _mm256_setzero_ps();
    __m256 acc2_b = _mm256_setzero_ps();
    __m256 acc3_a = _mm256_setzero_ps();
    __m256 acc3_b = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 q_lo = _mm256_loadu_ps(query + i);
        __m256 q_hi = _mm256_loadu_ps(query + i + 8);

        // Vector 0
        __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c0 + i));
        __m256 f0_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r0));
        __m256 f0_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(r0, 8)));
        __m256 val0_lo = _mm256_fmadd_ps(f0_lo, v_scale, v_min);
        __m256 val0_hi = _mm256_fmadd_ps(f0_hi, v_scale, v_min);
        __m256 diff0_lo = _mm256_sub_ps(q_lo, val0_lo);
        __m256 diff0_hi = _mm256_sub_ps(q_hi, val0_hi);
        acc0_a = _mm256_fmadd_ps(diff0_lo, diff0_lo, acc0_a);
        acc0_b = _mm256_fmadd_ps(diff0_hi, diff0_hi, acc0_b);

        // Vector 1
        __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c1 + i));
        __m256 f1_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r1));
        __m256 f1_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(r1, 8)));
        __m256 val1_lo = _mm256_fmadd_ps(f1_lo, v_scale, v_min);
        __m256 val1_hi = _mm256_fmadd_ps(f1_hi, v_scale, v_min);
        __m256 diff1_lo = _mm256_sub_ps(q_lo, val1_lo);
        __m256 diff1_hi = _mm256_sub_ps(q_hi, val1_hi);
        acc1_a = _mm256_fmadd_ps(diff1_lo, diff1_lo, acc1_a);
        acc1_b = _mm256_fmadd_ps(diff1_hi, diff1_hi, acc1_b);

        // Vector 2
        __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c2 + i));
        __m256 f2_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r2));
        __m256 f2_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(r2, 8)));
        __m256 val2_lo = _mm256_fmadd_ps(f2_lo, v_scale, v_min);
        __m256 val2_hi = _mm256_fmadd_ps(f2_hi, v_scale, v_min);
        __m256 diff2_lo = _mm256_sub_ps(q_lo, val2_lo);
        __m256 diff2_hi = _mm256_sub_ps(q_hi, val2_hi);
        acc2_a = _mm256_fmadd_ps(diff2_lo, diff2_lo, acc2_a);
        acc2_b = _mm256_fmadd_ps(diff2_hi, diff2_hi, acc2_b);

        // Vector 3
        __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c3 + i));
        __m256 f3_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r3));
        __m256 f3_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(r3, 8)));
        __m256 val3_lo = _mm256_fmadd_ps(f3_lo, v_scale, v_min);
        __m256 val3_hi = _mm256_fmadd_ps(f3_hi, v_scale, v_min);
        __m256 diff3_lo = _mm256_sub_ps(q_lo, val3_lo);
        __m256 diff3_hi = _mm256_sub_ps(q_hi, val3_hi);
        acc3_a = _mm256_fmadd_ps(diff3_lo, diff3_lo, acc3_a);
        acc3_b = _mm256_fmadd_ps(diff3_hi, diff3_hi, acc3_b);
    }

    for (; i + 7 < dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query + i);

        // Vector 0
        __m128i r0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0 + i));
        __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r0));
        __m256 val0 = _mm256_fmadd_ps(f0, v_scale, v_min);
        __m256 diff0 = _mm256_sub_ps(q, val0);
        acc0_a = _mm256_fmadd_ps(diff0, diff0, acc0_a);

        // Vector 1
        __m128i r1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1 + i));
        __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r1));
        __m256 val1 = _mm256_fmadd_ps(f1, v_scale, v_min);
        __m256 diff1 = _mm256_sub_ps(q, val1);
        acc1_a = _mm256_fmadd_ps(diff1, diff1, acc1_a);

        // Vector 2
        __m128i r2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2 + i));
        __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r2));
        __m256 val2 = _mm256_fmadd_ps(f2, v_scale, v_min);
        __m256 diff2 = _mm256_sub_ps(q, val2);
        acc2_a = _mm256_fmadd_ps(diff2, diff2, acc2_a);

        // Vector 3
        __m128i r3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3 + i));
        __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r3));
        __m256 val3 = _mm256_fmadd_ps(f3, v_scale, v_min);
        __m256 diff3 = _mm256_sub_ps(q, val3);
        acc3_a = _mm256_fmadd_ps(diff3, diff3, acc3_a);
    }

    float s0 = HorizontalSum(_mm256_add_ps(acc0_a, acc0_b));
    float s1 = HorizontalSum(_mm256_add_ps(acc1_a, acc1_b));
    float s2 = HorizontalSum(_mm256_add_ps(acc2_a, acc2_b));
    float s3 = HorizontalSum(_mm256_add_ps(acc3_a, acc3_b));

    for (; i < dim; ++i) {
        float q = query[i];
        float d0 = q - (min_val + static_cast<float>(c0[i]) * inv_scale);
        float d1 = q - (min_val + static_cast<float>(c1[i]) * inv_scale);
        float d2 = q - (min_val + static_cast<float>(c2[i]) * inv_scale);
        float d3 = q - (min_val + static_cast<float>(c3[i]) * inv_scale);
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    out_dists[0] = s0;
    out_dists[1] = s1;
    out_dists[2] = s2;
    out_dists[3] = s3;
}

__attribute__((target("avx2,fma")))
void DotF32_4x_Avx2(const float* query,
                    const float* v0, const float* v1, const float* v2, const float* v3,
                    size_t dim, float out_dots[4]) noexcept {
    // Dual-accumulator per vector (8 total): hides 4-cycle FMA latency,
    // processes 16 floats/iter instead of 8 — ~2x FMA pipeline utilization.
    __m256 acc0a = _mm256_setzero_ps(), acc0b = _mm256_setzero_ps();
    __m256 acc1a = _mm256_setzero_ps(), acc1b = _mm256_setzero_ps();
    __m256 acc2a = _mm256_setzero_ps(), acc2b = _mm256_setzero_ps();
    __m256 acc3a = _mm256_setzero_ps(), acc3b = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 q0 = _mm256_loadu_ps(query + i);
        __m256 q1 = _mm256_loadu_ps(query + i + 8);

        acc0a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(v0 + i),     acc0a);
        acc0b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(v0 + i + 8), acc0b);

        acc1a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(v1 + i),     acc1a);
        acc1b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(v1 + i + 8), acc1b);

        acc2a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(v2 + i),     acc2a);
        acc2b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(v2 + i + 8), acc2b);

        acc3a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(v3 + i),     acc3a);
        acc3b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(v3 + i + 8), acc3b);
    }
    // Fold second accumulator into first
    acc0a = _mm256_add_ps(acc0a, acc0b);
    acc1a = _mm256_add_ps(acc1a, acc1b);
    acc2a = _mm256_add_ps(acc2a, acc2b);
    acc3a = _mm256_add_ps(acc3a, acc3b);

    // 8-element tail
    for (; i + 7 < dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query + i);
        acc0a = _mm256_fmadd_ps(q, _mm256_loadu_ps(v0 + i), acc0a);
        acc1a = _mm256_fmadd_ps(q, _mm256_loadu_ps(v1 + i), acc1a);
        acc2a = _mm256_fmadd_ps(q, _mm256_loadu_ps(v2 + i), acc2a);
        acc3a = _mm256_fmadd_ps(q, _mm256_loadu_ps(v3 + i), acc3a);
    }

    float s0 = HorizontalSum(acc0a);
    float s1 = HorizontalSum(acc1a);
    float s2 = HorizontalSum(acc2a);
    float s3 = HorizontalSum(acc3a);

    for (; i < dim; ++i) {
        float q = query[i];
        s0 += q * v0[i];
        s1 += q * v1[i];
        s2 += q * v2[i];
        s3 += q * v3[i];
    }

    out_dots[0] = s0;
    out_dots[1] = s1;
    out_dots[2] = s2;
    out_dots[3] = s3;
}

__attribute__((target("avx2,fma")))
void L2SqF32_4x_Avx2(const float* query,
                     const float* v0, const float* v1, const float* v2, const float* v3,
                     size_t dim, float out_l2sq[4]) noexcept {
    // Dual-accumulator per vector (8 total): hides 4-cycle FMA latency,
    // processes 16 floats/iter instead of 8 — ~2x FMA pipeline utilization.
    __m256 acc0a = _mm256_setzero_ps(), acc0b = _mm256_setzero_ps();
    __m256 acc1a = _mm256_setzero_ps(), acc1b = _mm256_setzero_ps();
    __m256 acc2a = _mm256_setzero_ps(), acc2b = _mm256_setzero_ps();
    __m256 acc3a = _mm256_setzero_ps(), acc3b = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 q0 = _mm256_loadu_ps(query + i);
        __m256 q1 = _mm256_loadu_ps(query + i + 8);

        __m256 d0a = _mm256_sub_ps(q0, _mm256_loadu_ps(v0 + i));
        __m256 d0b = _mm256_sub_ps(q1, _mm256_loadu_ps(v0 + i + 8));
        acc0a = _mm256_fmadd_ps(d0a, d0a, acc0a);
        acc0b = _mm256_fmadd_ps(d0b, d0b, acc0b);

        __m256 d1a = _mm256_sub_ps(q0, _mm256_loadu_ps(v1 + i));
        __m256 d1b = _mm256_sub_ps(q1, _mm256_loadu_ps(v1 + i + 8));
        acc1a = _mm256_fmadd_ps(d1a, d1a, acc1a);
        acc1b = _mm256_fmadd_ps(d1b, d1b, acc1b);

        __m256 d2a = _mm256_sub_ps(q0, _mm256_loadu_ps(v2 + i));
        __m256 d2b = _mm256_sub_ps(q1, _mm256_loadu_ps(v2 + i + 8));
        acc2a = _mm256_fmadd_ps(d2a, d2a, acc2a);
        acc2b = _mm256_fmadd_ps(d2b, d2b, acc2b);

        __m256 d3a = _mm256_sub_ps(q0, _mm256_loadu_ps(v3 + i));
        __m256 d3b = _mm256_sub_ps(q1, _mm256_loadu_ps(v3 + i + 8));
        acc3a = _mm256_fmadd_ps(d3a, d3a, acc3a);
        acc3b = _mm256_fmadd_ps(d3b, d3b, acc3b);
    }
    // Fold second accumulator into first
    acc0a = _mm256_add_ps(acc0a, acc0b);
    acc1a = _mm256_add_ps(acc1a, acc1b);
    acc2a = _mm256_add_ps(acc2a, acc2b);
    acc3a = _mm256_add_ps(acc3a, acc3b);

    // 8-element tail
    for (; i + 7 < dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query + i);
        __m256 diff0 = _mm256_sub_ps(q, _mm256_loadu_ps(v0 + i));
        acc0a = _mm256_fmadd_ps(diff0, diff0, acc0a);
        __m256 diff1 = _mm256_sub_ps(q, _mm256_loadu_ps(v1 + i));
        acc1a = _mm256_fmadd_ps(diff1, diff1, acc1a);
        __m256 diff2 = _mm256_sub_ps(q, _mm256_loadu_ps(v2 + i));
        acc2a = _mm256_fmadd_ps(diff2, diff2, acc2a);
        __m256 diff3 = _mm256_sub_ps(q, _mm256_loadu_ps(v3 + i));
        acc3a = _mm256_fmadd_ps(diff3, diff3, acc3a);
    }

    float s0 = HorizontalSum(acc0a);
    float s1 = HorizontalSum(acc1a);
    float s2 = HorizontalSum(acc2a);
    float s3 = HorizontalSum(acc3a);

    for (; i < dim; ++i) {
        float q = query[i];
        float d0 = q - v0[i];
        float d1 = q - v1[i];
        float d2 = q - v2[i];
        float d3 = q - v3[i];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    out_l2sq[0] = s0;
    out_l2sq[1] = s1;
    out_l2sq[2] = s2;
    out_l2sq[3] = s3;
}

#endif // AVX2 check

inline bool HasAvx2Fma() noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

// AVX2 horizontal sum: dual-acc (acc0 low, acc1 high) to hide 4-cycle add latency.
// 16 floats/iter — ~16x faster than scalar chain for dim=128.
__attribute__((target("avx2,fma")))
float SumF32_Avx2(const float* data, size_t dim) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(data + i + 8));
    }
    acc0 = _mm256_add_ps(acc0, acc1); // fold
    for (; i + 7 < dim; i += 8) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
    }
    float s = HorizontalSum(acc0);
    for (; i < dim; ++i) s += data[i];
    return s;
}

} // namespace

void DotSq8_4x(const float* query,
               const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3,
               size_t dim, float min_val, float inv_scale, float query_sum,
               float out_scores[4]) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    static const bool has_avx2 = HasAvx2Fma();
    if (has_avx2) {
        DotSq8_4x_Avx2(query, c0, c1, c2, c3, dim, min_val, inv_scale, query_sum, out_scores);
        return;
    }
#endif
    DotSq8_4x_Scalar(query, c0, c1, c2, c3, dim, min_val, inv_scale, query_sum, out_scores);
}

void L2SqSq8_4x(const float* query,
                const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3,
                size_t dim, float min_val, float max_val,
                float out_dists[4]) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    static const bool has_avx2 = HasAvx2Fma();
    if (has_avx2) {
        L2SqSq8_4x_Avx2(query, c0, c1, c2, c3, dim, min_val, max_val, out_dists);
        return;
    }
#endif
    L2SqSq8_4x_Scalar(query, c0, c1, c2, c3, dim, min_val, max_val, out_dists);
}

void DotF32_4x(const float* query,
               const float* v0, const float* v1, const float* v2, const float* v3,
               size_t dim, float out_dots[4]) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    static const bool has_avx2 = HasAvx2Fma();
    if (has_avx2) {
        DotF32_4x_Avx2(query, v0, v1, v2, v3, dim, out_dots);
        return;
    }
#endif
    DotF32_4x_Scalar(query, v0, v1, v2, v3, dim, out_dots);
}

void L2SqF32_4x(const float* query,
                const float* v0, const float* v1, const float* v2, const float* v3,
                size_t dim, float out_l2sq[4]) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    static const bool has_avx2 = HasAvx2Fma();
    if (has_avx2) {
        L2SqF32_4x_Avx2(query, v0, v1, v2, v3, dim, out_l2sq);
        return;
    }
#endif
    L2SqF32_4x_Scalar(query, v0, v1, v2, v3, dim, out_l2sq);
}

float SumF32(const float* data, size_t dim) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    static const bool has_avx2 = HasAvx2Fma();
    if (has_avx2) return SumF32_Avx2(data, dim);
#endif
    return SumF32_Scalar(data, dim);
}

void PulpBatchScanner::Scan4(const float* query, size_t dim,
                            MetricType metric, float query_sum,
                            const storage::PulpView& pulp,
                            uint32_t slot_base, uint32_t valid_count,
                            float scores_out[4]) noexcept {
    if (valid_count == 0) {
        scores_out[0] = scores_out[1] = scores_out[2] = scores_out[3] = -1e30f;
        return;
    }

    const bool is_ip = (metric == MetricType::kInnerProduct || metric == MetricType::kCosine);

    if (valid_count == 4) {
        const uint8_t* c0 = pulp.GetCodes(slot_base).data();
        const uint8_t* c1 = pulp.GetCodes(slot_base + 1).data();
        const uint8_t* c2 = pulp.GetCodes(slot_base + 2).data();
        const uint8_t* c3 = pulp.GetCodes(slot_base + 3).data();

        if (is_ip) {
            DotSq8_4x(query, c0, c1, c2, c3, dim, pulp.quant_min(), pulp.quant_inv_scale(), query_sum, scores_out);
        } else {
            float dists[4];
            const float max_val = pulp.quant_min() + 255.0f * pulp.quant_inv_scale();
            L2SqSq8_4x(query, c0, c1, c2, c3, dim, pulp.quant_min(), max_val, dists);
            scores_out[0] = -dists[0];
            scores_out[1] = -dists[1];
            scores_out[2] = -dists[2];
            scores_out[3] = -dists[3];
        }
    } else {
        std::span<const float> q_span(query, dim);
        for (uint32_t i = 0; i < valid_count; ++i) {
            scores_out[i] = pulp.Taste(q_span, slot_base + i, metric, query_sum);
        }
        for (uint32_t i = valid_count; i < 4; ++i) {
            scores_out[i] = -1e30f;
        }
    }
}

void SeedBatchReranker::Rerank4(const float* query, size_t dim,
                               MetricType metric,
                               const float* v0, const float* v1, const float* v2, const float* v3,
                               float scores_out[4]) noexcept {
    if (metric == MetricType::kInnerProduct) {
        DotF32_4x(query, v0, v1, v2, v3, dim, scores_out);
    } else if (metric == MetricType::kL2) {
        float l2sq[4];
        L2SqF32_4x(query, v0, v1, v2, v3, dim, l2sq);
        scores_out[0] = -l2sq[0];
        scores_out[1] = -l2sq[1];
        scores_out[2] = -l2sq[2];
        scores_out[3] = -l2sq[3];
    } else { // Cosine
        float dots[4];
        DotF32_4x(query, v0, v1, v2, v3, dim, dots);
        double q_norm_sq = 0.0;
        double n0 = 0.0, n1 = 0.0, n2 = 0.0, n3 = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            double q = static_cast<double>(query[i]);
            q_norm_sq += q * q;
            n0 += static_cast<double>(v0[i]) * static_cast<double>(v0[i]);
            n1 += static_cast<double>(v1[i]) * static_cast<double>(v1[i]);
            n2 += static_cast<double>(v2[i]) * static_cast<double>(v2[i]);
            n3 += static_cast<double>(v3[i]) * static_cast<double>(v3[i]);
        }
        double q_norm = std::sqrt(q_norm_sq);
        auto calc_cos = [&](float dot, double norm) -> float {
            if (q_norm <= 1e-12 || norm <= 1e-12) return 0.0f;
            double cos_val = static_cast<double>(dot) / (q_norm * std::sqrt(norm));
            return static_cast<float>(std::clamp(cos_val, -1.0, 1.0));
        };
        scores_out[0] = calc_cos(dots[0], n0);
        scores_out[1] = calc_cos(dots[1], n1);
        scores_out[2] = calc_cos(dots[2], n2);
        scores_out[3] = calc_cos(dots[3], n3);
    }
}

} // namespace pomai::compute
