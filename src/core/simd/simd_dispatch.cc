#include "core/simd/simd_dispatch.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pomai::core::simd {

namespace {
constexpr double kEarthRadiusMeters = 6371000.0;
}

double HaversineMeters(double lat1, double lon1, double lat2, double lon2) {
    const double dlat = (lat2 - lat1) * M_PI / 180.0;
    const double dlon = (lon2 - lon1) * M_PI / 180.0;
    const double a1 = lat1 * M_PI / 180.0;
    const double a2 = lat2 * M_PI / 180.0;
    const double h = std::sin(dlat * 0.5) * std::sin(dlat * 0.5) +
                     std::cos(a1) * std::cos(a2) *
                     std::sin(dlon * 0.5) * std::sin(dlon * 0.5);
    const double c = 2.0 * std::atan2(std::sqrt(h), std::sqrt(1.0 - h));
    return kEarthRadiusMeters * c;
}

double MeshRmsdF32(const float* a_xyz, const float* b_xyz, std::size_t points) {
    if (!a_xyz || !b_xyz || points == 0) return 0.0;
    double sum = 0.0;
    const std::size_t n = points * 3;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a_xyz[i]) - static_cast<double>(b_xyz[i]);
        sum += d * d;
    }
    return std::sqrt(sum / static_cast<double>(n));
}

double SparseDotU32F32(const std::uint32_t* a_idx, const float* a_w, std::size_t a_n,
                       const std::uint32_t* b_idx, const float* b_w, std::size_t b_n) {
    if (!a_idx || !b_idx || !a_w || !b_w || a_n == 0 || b_n == 0) return 0.0;
    std::size_t i = 0, j = 0;
    double sum = 0.0;
    while (i < a_n && j < b_n) {
        if (a_idx[i] == b_idx[j]) {
            sum += static_cast<double>(a_w[i]) * static_cast<double>(b_w[j]);
            ++i;
            ++j;
        } else if (a_idx[i] < b_idx[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return sum;
}

std::uint32_t SparseIntersectU32(const std::uint32_t* a_idx, std::size_t a_n,
                                 const std::uint32_t* b_idx, std::size_t b_n) {
    if (!a_idx || !b_idx || a_n == 0 || b_n == 0) return 0;
    std::size_t i = 0, j = 0;
    std::uint32_t inter = 0;
    while (i < a_n && j < b_n) {
        if (a_idx[i] == b_idx[j]) {
            ++inter;
            ++i;
            ++j;
        } else if (a_idx[i] < b_idx[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return inter;
}

double BitsetHamming(const std::uint8_t* a, const std::uint8_t* b, std::size_t n_bytes) {
    if (!a || !b || n_bytes == 0) return 0.0;
    std::uint64_t diff = 0;
    for (std::size_t i = 0; i < n_bytes; ++i) {
        diff += static_cast<std::uint64_t>(__builtin_popcount(static_cast<unsigned int>(a[i] ^ b[i])));
    }
    return static_cast<double>(diff);
}

double BitsetJaccard(const std::uint8_t* a, const std::uint8_t* b, std::size_t n_bytes) {
    if (!a || !b || n_bytes == 0) return 1.0;
    std::uint64_t inter = 0;
    std::uint64_t uni = 0;
    for (std::size_t i = 0; i < n_bytes; ++i) {
        inter += static_cast<std::uint64_t>(__builtin_popcount(static_cast<unsigned int>(a[i] & b[i])));
        uni += static_cast<std::uint64_t>(__builtin_popcount(static_cast<unsigned int>(a[i] | b[i])));
    }
    if (uni == 0) return 1.0;
    return static_cast<double>(inter) / static_cast<double>(uni);
}

} // namespace pomai::core::simd

