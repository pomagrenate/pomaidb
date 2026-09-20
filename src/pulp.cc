// pomai/pulp.cc — Fast approximate representation (Pulp) implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "pulp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#define POMAI_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define POMAI_TARGET_AVX2
#endif
#endif

namespace pomai::storage {
namespace {

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
static inline bool CpuHasAvx2() {
    return __builtin_cpu_supports("avx2");
}

POMAI_TARGET_AVX2
static void VectorMinMaxAvx2(const float* vec, size_t dim, float& g_min, float& g_max, bool& found_valid) {
    size_t i = 0;
    __m256 v_min = _mm256_set1_ps(g_min);
    __m256 v_max = _mm256_set1_ps(g_max);

    for (; i + 7 < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        v_min = _mm256_min_ps(v_min, v);
        v_max = _mm256_max_ps(v_max, v);
        found_valid = true;
    }

    alignas(32) float min_buf[8];
    alignas(32) float max_buf[8];
    _mm256_storeu_ps(min_buf, v_min);
    _mm256_storeu_ps(max_buf, v_max);

    for (int k = 0; k < 8; ++k) {
        if (std::isfinite(min_buf[k]) && min_buf[k] < g_min) g_min = min_buf[k];
        if (std::isfinite(max_buf[k]) && max_buf[k] > g_max) g_max = max_buf[k];
    }

    for (; i < dim; ++i) {
        float f = vec[i];
        if (std::isfinite(f)) {
            if (f < g_min) g_min = f;
            if (f > g_max) g_max = f;
            found_valid = true;
        }
    }
}

POMAI_TARGET_AVX2
static void EncodeAppendAvx2(const float* vec, size_t dim, float min_val, float scale, uint8_t* dst) {
    size_t i = 0;
    __m256 v_min = _mm256_set1_ps(min_val);
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_255 = _mm256_set1_ps(255.0f);

    for (; i + 7 < dim; i += 8) {
        __m256 x = _mm256_loadu_ps(vec + i);
        __m256 n = _mm256_mul_ps(_mm256_sub_ps(x, v_min), v_scale);
        __m256 c = _mm256_min_ps(_mm256_max_ps(n, v_zero), v_255);
        __m256 r = _mm256_round_ps(c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i i0 = _mm256_cvtps_epi32(r);

        __m128i lo = _mm256_castsi256_si128(i0);
        __m128i hi = _mm256_extracti128_si256(i0, 1);
        __m128i p16 = _mm_packus_epi32(lo, hi);
        __m128i p8 = _mm_packus_epi16(p16, p16);

        uint64_t bytes8 = static_cast<uint64_t>(_mm_cvtsi128_si64(p8));
        std::memcpy(dst + i, &bytes8, sizeof(uint64_t));
    }

    for (; i < dim; ++i) {
        float normalized = (vec[i] - min_val) * scale;
        float clamped = std::clamp(normalized, 0.0f, 255.0f);
        dst[i] = static_cast<uint8_t>(std::round(clamped));
    }
}
#endif

} // namespace

PulpBuilder::PulpBuilder(uint32_t dim, uint8_t quant_type)
    : dim_(dim), count_(0), quant_type_(quant_type),
      min_val_(0.0f), scale_(0.0f), inv_scale_(0.0f) {}

void PulpBuilder::Train(const std::vector<std::span<const float>>& vectors) {
    if (vectors.empty() || dim_ == 0) return;

    float g_min = std::numeric_limits<float>::max();
    float g_max = std::numeric_limits<float>::lowest();
    bool found_valid = false;

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    if (CpuHasAvx2()) {
        for (const auto& v : vectors) {
            if (v.size() != dim_) continue;
            VectorMinMaxAvx2(v.data(), dim_, g_min, g_max, found_valid);
        }
    } else
#endif
    {
        for (const auto& v : vectors) {
            if (v.size() != dim_) continue;
            for (float f : v) {
                if (std::isfinite(f)) {
                    if (f < g_min) g_min = f;
                    if (f > g_max) g_max = f;
                    found_valid = true;
                }
            }
        }
    }

    if (!found_valid) {
        min_val_ = 0.0f;
        scale_ = 0.0f;
        inv_scale_ = 0.0f;
        return;
    }

    min_val_ = g_min;
    const float range = g_max - g_min;
    if (range <= 1e-6f) {
        scale_ = 0.0f;
        inv_scale_ = 0.0f;
    } else {
        scale_ = 255.0f / range;
        inv_scale_ = range / 255.0f;
    }
}

void PulpBuilder::EncodeAppend(std::span<const float> vec) {
    if (vec.size() != dim_) return;

    const size_t prev_size = buffer_.size();
    buffer_.resize(prev_size + dim_);
    uint8_t* dst = buffer_.data() + prev_size;

    if (scale_ <= 0.0f) {
        std::memset(dst, 0, dim_);
    } else {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
        if (CpuHasAvx2()) {
            EncodeAppendAvx2(vec.data(), dim_, min_val_, scale_, dst);
        } else
#endif
        {
            for (size_t i = 0; i < dim_; ++i) {
                float normalized = (vec[i] - min_val_) * scale_;
                float clamped = std::clamp(normalized, 0.0f, 255.0f);
                dst[i] = static_cast<uint8_t>(std::round(clamped));
            }
        }
    }
    ++count_;
}

} // namespace pomai::storage
