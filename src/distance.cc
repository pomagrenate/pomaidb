// distance.cc — Standardized SIMD distance kernels with dynamic CPU dispatch
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "distance.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <psync/psync.h>
#include <vector>

#include "utils/half_float.h"
#include "utils/scratch_buffer.h"

// SimSIMD: by default uses compile-time dispatch.
// For portable binaries with runtime dispatch, we query simsimd_capabilities()
// and simsimd_find_kernel_punned at initialization.
#if !((defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) && defined(__ARM_FP16_FORMAT_IEEE)) && \
    !(((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && defined(__AVX512FP16__)))
#ifndef SIMSIMD_NATIVE_F16
#define SIMSIMD_NATIVE_F16 0
#endif
#endif

#if !((defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) && defined(__ARM_BF16_FORMAT_ALTERNATIVE)) && \
    !(((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && defined(__AVX512BF16__)))
#ifndef SIMSIMD_NATIVE_BF16
#define SIMSIMD_NATIVE_BF16 0
#endif
#endif
#include "simd/simsimd.h"

namespace pomai::core {
namespace {

simsimd_metric_dense_punned_t g_dot_f32 = nullptr;
simsimd_metric_dense_punned_t g_l2sq_f32 = nullptr;
psync::OnceFlag g_init_flag;

void InitOnce() {
    simsimd_capability_t cap = simsimd_capabilities();
    simsimd_kernel_punned_t k_dot = nullptr;
    simsimd_kernel_punned_t k_l2sq = nullptr;
    simsimd_capability_t c_dot = simsimd_cap_serial_k;
    simsimd_capability_t c_l2sq = simsimd_cap_serial_k;

    simsimd_find_kernel_punned(simsimd_metric_dot_k, simsimd_datatype_f32_k, cap, simsimd_cap_any_k, &k_dot, &c_dot);
    simsimd_find_kernel_punned(simsimd_metric_l2sq_k, simsimd_datatype_f32_k, cap, simsimd_cap_any_k, &k_l2sq, &c_l2sq);

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
    g_dot_f32 = reinterpret_cast<simsimd_metric_dense_punned_t>(k_dot);
    g_l2sq_f32 = reinterpret_cast<simsimd_metric_dense_punned_t>(k_l2sq);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

inline void EnsureInit() {
    psync::call_once(g_init_flag, InitOnce);
}

// ── Scalar fallback for DotSq8 (no SimSIMD equivalent) ──
float DotSq8Scalar(std::span<const float> q, std::span<const uint8_t> c,
                   float min_val, float inv_scale, float q_sum) {
    if (q.empty() || c.empty() || q.size() != c.size()) return 0.0f;
    float sum = 0.0f;
    for (std::size_t i = 0; i < q.size(); ++i)
        sum += q[i] * static_cast<float>(c[i]);
    return sum * inv_scale + q_sum * min_val;
}

// ── Binary Quantization (1-bit; optimized specifically for edge memory reduction) ──
void BitQuantizeImpl(std::span<const float> vec, uint8_t* out_codes) {
    if (!out_codes || vec.empty()) return;
    std::uint32_t dim = static_cast<std::uint32_t>(vec.size());
    std::uint32_t byte_count = (dim + 7) / 8;
    std::memset(out_codes, 0, byte_count);
    for (uint32_t i = 0; i < dim; ++i) {
        if (vec[i] > 0.0f) {
            out_codes[i / 8] |= (1u << (i % 8));
        }
    }
}

float HammingDistImpl(std::span<const uint8_t> a, std::span<const uint8_t> b) {
    uint32_t dist = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        dist += static_cast<uint32_t>(__builtin_popcount(a[i] ^ b[i]));
    }
    return static_cast<float>(dist);
}

float DotBitImpl(std::span<const float> query, std::span<const uint8_t> codes) {
    if (query.empty() || codes.empty()) return 0.0f;
    std::vector<uint8_t> q_codes((query.size() + 7) / 8);
    BitQuantizeImpl(query, q_codes.data());
    return static_cast<float>(query.size()) - HammingDistImpl(q_codes, codes);
}

// ── F32: use dynamically dispatched SimSIMD ──
float DotSimSIMD(std::span<const float> a, std::span<const float> b) {
    const std::size_t n = a.size();
    if (n == 0 || b.size() != n) return 0.0f;
    EnsureInit();
    simsimd_distance_t d = 0.0;
    if (g_dot_f32) {
        g_dot_f32(a.data(), b.data(), static_cast<simsimd_size_t>(n), &d);
    } else {
        simsimd_dot_f32(a.data(), b.data(), static_cast<simsimd_size_t>(n), &d);
    }
    return static_cast<float>(d);
}

float L2SqSimSIMD(std::span<const float> a, std::span<const float> b) {
    const std::size_t n = a.size();
    if (n == 0 || b.size() != n) return 0.0f;
    EnsureInit();
    simsimd_distance_t d = 0.0;
    if (g_l2sq_f32) {
        g_l2sq_f32(a.data(), b.data(), static_cast<simsimd_size_t>(n), &d);
    } else {
        simsimd_l2sq_f32(a.data(), b.data(), static_cast<simsimd_size_t>(n), &d);
    }
    return static_cast<float>(d);
}

// FP16: SimSIMD f16 (query converted f32->f16).
float DotFp16SimSIMD(std::span<const float> q, std::span<const uint16_t> c) {
    const std::size_t n = q.size();
    if (n == 0 || c.size() != n) return 0.0f;
    
    // CRITICAL FIX: Use thread-local scratch buffer instead of heap allocation
    simsimd_f16_t* q_f16 = reinterpret_cast<simsimd_f16_t*>(
        util::UInt16Scratch::Get(n));
    
    for (std::size_t i = 0; i < n; ++i)
        simsimd_f32_to_f16(q[i], &q_f16[i]);
    
    simsimd_distance_t d = 0.0;
    simsimd_dot_f16(q_f16, reinterpret_cast<const simsimd_f16_t*>(c.data()),
                    static_cast<simsimd_size_t>(n), &d);
    return static_cast<float>(d);
}

float L2SqFp16SimSIMD(std::span<const float> q, std::span<const uint16_t> c) {
    const std::size_t n = q.size();
    if (n == 0 || c.size() != n) return 0.0f;
    // CRITICAL FIX: Use thread-local scratch buffer instead of heap allocation
    simsimd_f16_t* q_f16 = reinterpret_cast<simsimd_f16_t*>(
        util::UInt16Scratch::Get(n));
    for (std::size_t i = 0; i < n; ++i)
        simsimd_f32_to_f16(q[i], &q_f16[i]);
    simsimd_distance_t d = 0.0;
    simsimd_l2sq_f16(q_f16, reinterpret_cast<const simsimd_f16_t*>(c.data()),
                     static_cast<simsimd_size_t>(n), &d);
    return static_cast<float>(d);
}

constexpr std::size_t kSq8Chunk = 32u;

}  // namespace

void InitDistance() {
    EnsureInit();
}

float Dot(std::span<const float> a, std::span<const float> b) {
    return DotSimSIMD(a, b);
}

float L2Sq(std::span<const float> a, std::span<const float> b) {
    return L2SqSimSIMD(a, b);
}

float L2(std::span<const float> a, std::span<const float> b) {
    float sq = L2Sq(a, b);
    return sq > 0.0f ? std::sqrt(sq) : 0.0f;
}

float CosineSimilarity(std::span<const float> a, std::span<const float> b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        const double da = static_cast<double>(a[i]);
        const double db = static_cast<double>(b[i]);
        dot += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }
    const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom <= 1e-12) return 0.0f;
    float sim = static_cast<float>(dot / denom);
    return std::clamp(sim, -1.0f, 1.0f);
}

float CosineDistance(std::span<const float> a, std::span<const float> b) {
    return 1.0f - CosineSimilarity(a, b);
}

float ComputeMetricScore(MetricType metric, std::span<const float> query, std::span<const float> vec) {
    switch (metric) {
        case MetricType::kL2:
            return -L2Sq(query, vec);
        case MetricType::kCosine:
            return CosineSimilarity(query, vec);
        case MetricType::kInnerProduct:
        default:
            return Dot(query, vec);
    }
}

float DotSq8(std::span<const float> q, std::span<const uint8_t> c,
             float min_val, float inv_scale, float q_sum) {
    return DotSq8Scalar(q, c, min_val, inv_scale, q_sum);
}

float L2SqSq8(std::span<const float> query,
              std::span<const std::uint8_t> data,
              float min_val, float max_val) {
    const std::size_t n = query.size();
    if (n == 0 || data.size() != n) return 0.0f;
    const float inv_scale = (max_val - min_val <= 1e-9f) ? 0.0f : ((max_val - min_val) / 255.0f);
    float sum_sq = 0.0f;
    std::size_t i = 0;
    float chunk_buf[kSq8Chunk];
    for (; i + kSq8Chunk <= n; i += kSq8Chunk) {
        for (std::size_t j = 0; j < kSq8Chunk; ++j)
            chunk_buf[j] = min_val + static_cast<float>(data[i + j]) * inv_scale;
        sum_sq += L2Sq(query.subspan(i, kSq8Chunk), std::span<const float>(chunk_buf, kSq8Chunk));
    }
    if (i < n) {
        const std::size_t rem = n - i;
        for (std::size_t j = 0; j < rem; ++j)
            chunk_buf[j] = min_val + static_cast<float>(data[i + j]) * inv_scale;
        sum_sq += L2Sq(query.subspan(i, rem), std::span<const float>(chunk_buf, rem));
    }
    return sum_sq;
}

float DotFp16(std::span<const float> q, std::span<const uint16_t> c) {
    return DotFp16SimSIMD(q, c);
}

float L2SqFp16(std::span<const float> q, std::span<const uint16_t> c) {
    return L2SqFp16SimSIMD(q, c);
}

void DotBatch(std::span<const float> query,
              const float* db, std::size_t n, std::uint32_t dim,
              float* results) {
    if (!db || !results || dim == 0) return;
    for (std::size_t i = 0; i < n; ++i) {
        results[i] = Dot(query, std::span<const float>(db + i * dim, dim));
    }
}

void L2SqBatch(std::span<const float> query,
               const float* db, std::size_t n, std::uint32_t dim,
               float* results) {
    if (!db || !results || dim == 0) return;
    for (std::size_t i = 0; i < n; ++i) {
        results[i] = L2Sq(query, std::span<const float>(db + i * dim, dim));
    }
}

void SearchBatch(std::span<const float> query, const FloatBatch& batch,
                 DistanceMetrics metric, float* results) {
    if (batch.format() == VectorFormat::FLAT) {
        if (metric == DistanceMetrics::DOT)
            DotBatch(query, batch.data(), batch.size(), batch.dim(), results);
        else
            L2SqBatch(query, batch.data(), batch.size(), batch.dim(), results);
    } else if (batch.format() == VectorFormat::DICTIONARY) {
        const std::uint32_t* sel = batch.selection();
        for (std::uint32_t i = 0; i < batch.size(); ++i) {
            const float* v = batch.get_vector(sel[i]);
            if (metric == DistanceMetrics::DOT)
                results[i] = Dot(query, std::span<const float>(v, batch.dim()));
            else
                results[i] = L2Sq(query, std::span<const float>(v, batch.dim()));
        }
    }
}

void BitQuantize(std::span<const float> vec, uint8_t* out_codes) {
    BitQuantizeImpl(vec, out_codes);
}

float HammingDist(std::span<const uint8_t> a, std::span<const uint8_t> b) {
    return HammingDistImpl(a, b);
}

float DotBit(std::span<const float> query, std::span<const uint8_t> codes) {
    return DotBitImpl(query, codes);
}

}  // namespace pomai::core
