// include/pvec/pvec_quant_fp16.h — Half-Precision (FP16 & BF16) Quantization
// Copyright 2026 PomaiDB / pvec authors. MIT License.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include "pvec_platform.h"

namespace pvec {

// ── FP16 / BF16 Bit-Level Conversions ────────────────────────────────────────

inline uint16_t float_to_fp16(float f) noexcept {
#if defined(PVEC_ARCH_X86_64) && defined(__F16C__)
    return static_cast<uint16_t>(_cvtss_sh(f, 0));
#else
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x0FFF + ((mant >> 13) & 1)) >> 13));
    } else if (exp == 0xFF - 127 + 15) {
        if (mant == 0) return static_cast<uint16_t>(sign | 0x7C00); // Inf
        return static_cast<uint16_t>(sign | 0x7E00 | (mant >> 13)); // NaN
    }
    if (exp > 30) return static_cast<uint16_t>(sign | 0x7C00); // Overflow -> Inf
    return static_cast<uint16_t>(sign | (exp << 10) | ((mant + 0x0FFF + ((mant >> 13) & 1)) >> 13));
#endif
}

inline float fp16_to_float(uint16_t h) noexcept {
#if defined(PVEC_ARCH_X86_64) && defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;

    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // Subnormal
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000 | (mant << 13); // Inf or NaN
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(float));
    return f;
#endif
}

inline uint16_t float_to_bf16(float f) noexcept {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    // Round to nearest even
    uint32_t lsb = (x >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    x += rounding_bias;
    return static_cast<uint16_t>(x >> 16);
}

inline float bf16_to_float(uint16_t b) noexcept {
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(float));
    return f;
}

// ── FP16 Batch Conversion & Distance ─────────────────────────────────────────

class HalfFloatQuantizer {
public:
    static void Encode(std::span<const float> in, std::span<uint16_t> out) noexcept {
        const std::size_t n = std::min(in.size(), out.size());
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = float_to_fp16(in[i]);
        }
    }

    static void Decode(std::span<const uint16_t> in, std::span<float> out) noexcept {
        const std::size_t n = std::min(in.size(), out.size());
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = fp16_to_float(in[i]);
        }
    }

    /// Asymmetric Dot Product: Float Query vs FP16 Codes
    static float Dot(std::span<const float> query, std::span<const uint16_t> codes) noexcept {
        const std::size_t n = std::min(query.size(), codes.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += static_cast<double>(query[i]) * static_cast<double>(fp16_to_float(codes[i]));
        }
        return static_cast<float>(sum);
    }

    /// Asymmetric Squared L2 Distance: Float Query vs FP16 Codes
    static float L2Sq(std::span<const float> query, std::span<const uint16_t> codes) noexcept {
        const std::size_t n = std::min(query.size(), codes.size());
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double d = static_cast<double>(query[i]) - static_cast<double>(fp16_to_float(codes[i]));
            sum_sq += d * d;
        }
        return static_cast<float>(sum_sq);
    }
};

} // namespace pvec
