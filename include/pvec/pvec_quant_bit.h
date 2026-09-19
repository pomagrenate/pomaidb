// include/pvec/pvec_quant_bit.h — 1-Bit Binary Quantization & Popcount Hamming Distance
// Copyright 2026 PomaiDB / pvec authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <algorithm>
#include "pvec_platform.h"

namespace pvec {

class BitQuantizer {
public:
    /// Encodes float vector into 1-bit binary codes (1 if v > 0, 0 otherwise).
    /// Output buffer must have size >= (vec.size() + 7) / 8 bytes.
    static void Encode(std::span<const float> vec, uint8_t* out_codes) noexcept {
        if (!out_codes || vec.empty()) return;
        const std::size_t dim = vec.size();
        const std::size_t byte_count = (dim + 7) / 8;
        std::memset(out_codes, 0, byte_count);

        for (std::size_t i = 0; i < dim; ++i) {
            if (vec[i] > 0.0f) {
                out_codes[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            }
        }
    }

    /// Fast 64-bit hardware popcount Hamming Distance between two binary code spans
    static uint32_t HammingDistance(std::span<const uint8_t> a, std::span<const uint8_t> b) noexcept {
        uint32_t dist = 0;
        const std::size_t n = std::min(a.size(), b.size());
        std::size_t i = 0;

        for (; i + 7 < n; i += 8) {
            uint64_t val_a, val_b;
            std::memcpy(&val_a, a.data() + i, sizeof(uint64_t));
            std::memcpy(&val_b, b.data() + i, sizeof(uint64_t));
#if defined(_MSC_VER) && !defined(__clang__)
            dist += static_cast<uint32_t>(__popcnt64(val_a ^ val_b));
#else
            dist += static_cast<uint32_t>(__builtin_popcountll(val_a ^ val_b));
#endif
        }

        for (; i < n; ++i) {
            dist += static_cast<uint32_t>(__builtin_popcount(a[i] ^ b[i]));
        }
        return dist;
    }

    /// Binary Dot Product Score (Higher = More similar, range [-dim, +dim])
    /// score = dim - 2 * hamming_distance
    static float DotBit(std::span<const uint8_t> a_codes,
                       std::span<const uint8_t> b_codes,
                       std::size_t dimension) noexcept {
        uint32_t h = HammingDistance(a_codes, b_codes);
        return static_cast<float>(dimension) - 2.0f * static_cast<float>(h);
    }
};

} // namespace pvec
