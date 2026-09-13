// reference_distance.cc — Exact, obvious scalar reference implementation (oracle)
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "reference_distance.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#include "utils/half_float.h"

namespace pomai::core::reference {

double L2Sq(std::span<const float> a, std::span<const float> b) noexcept {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0;
    }
    double sum = 0.0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return sum;
}

double L2(std::span<const float> a, std::span<const float> b) noexcept {
    return std::sqrt(L2Sq(a, b));
}

double InnerProduct(std::span<const float> a, std::span<const float> b) noexcept {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0;
    }
    double sum = 0.0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

double CosineSimilarity(std::span<const float> a, std::span<const float> b) noexcept {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0;
    }
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
    if (denom <= 1e-12) {
        return 0.0; // Contract: zero-norm vector yields 0.0
    }
    double cos_sim = dot / denom;
    // Bound to mathematical [-1.0, 1.0] against precision overflow
    return std::clamp(cos_sim, -1.0, 1.0);
}

double CosineDistance(std::span<const float> a, std::span<const float> b) noexcept {
    return 1.0 - CosineSimilarity(a, b);
}

double DotSq8(std::span<const float> query,
              std::span<const uint8_t> codes,
              float min_val, float inv_scale, float query_sum) noexcept {
    if (query.empty() || codes.empty() || query.size() != codes.size()) {
        return 0.0;
    }
    double sum = 0.0;
    const size_t n = query.size();
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(query[i]) * static_cast<double>(codes[i]);
    }
    return sum * static_cast<double>(inv_scale) + static_cast<double>(query_sum) * static_cast<double>(min_val);
}

double L2SqSq8(std::span<const float> query,
               std::span<const uint8_t> codes,
               float min_val, float max_val) noexcept {
    if (query.empty() || codes.empty() || query.size() != codes.size()) {
        return 0.0;
    }
    const double range = static_cast<double>(max_val) - static_cast<double>(min_val);
    const double inv_scale = (range <= 1e-9) ? 0.0 : (range / 255.0);
    double sum_sq = 0.0;
    const size_t n = query.size();
    for (size_t i = 0; i < n; ++i) {
        const double dequant = static_cast<double>(min_val) + static_cast<double>(codes[i]) * inv_scale;
        const double diff = static_cast<double>(query[i]) - dequant;
        sum_sq += diff * diff;
    }
    return sum_sq;
}

double DotFp16(std::span<const float> query,
               std::span<const uint16_t> codes) noexcept {
    if (query.empty() || codes.empty() || query.size() != codes.size()) {
        return 0.0;
    }
    double sum = 0.0;
    const size_t n = query.size();
    for (size_t i = 0; i < n; ++i) {
        const float f = pomai::util::float16_to_float32(codes[i]);
        sum += static_cast<double>(query[i]) * static_cast<double>(f);
    }
    return sum;
}

double L2SqFp16(std::span<const float> query,
                std::span<const uint16_t> codes) noexcept {
    if (query.empty() || codes.empty() || query.size() != codes.size()) {
        return 0.0;
    }
    double sum_sq = 0.0;
    const size_t n = query.size();
    for (size_t i = 0; i < n; ++i) {
        const float f = pomai::util::float16_to_float32(codes[i]);
        const double diff = static_cast<double>(query[i]) - static_cast<double>(f);
        sum_sq += diff * diff;
    }
    return sum_sq;
}

uint32_t HammingDist(std::span<const uint8_t> a, std::span<const uint8_t> b) noexcept {
    const size_t n = std::min(a.size(), b.size());
    uint32_t dist = 0;
    for (size_t i = 0; i < n; ++i) {
        uint8_t diff = a[i] ^ b[i];
        while (diff > 0) {
            dist += (diff & 1u);
            diff >>= 1;
        }
    }
    return dist;
}

void BitQuantize(std::span<const float> vec, uint8_t* out_codes) noexcept {
    if (!out_codes || vec.empty()) return;
    const size_t byte_count = (vec.size() + 7) / 8;
    std::memset(out_codes, 0, byte_count);
    for (size_t i = 0; i < vec.size(); ++i) {
        if (vec[i] > 0.0f) {
            out_codes[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
    }
}

} // namespace pomai::core::reference
