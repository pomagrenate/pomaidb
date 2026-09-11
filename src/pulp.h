// pomai/pulp.h — Fast approximate representation (Pulp) for PomaiDB Arils
//
// In the Pomegranate Engine:
// - Pulp is the easily accessed flesh: compressed (SQ8 / BQ / FP16), compact, contiguous,
//   64-byte aligned, SIMD-friendly.
// - Taste: Approximate scoring scans Pulp without reading expensive Seed Kernels (FP32).
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "distance.h"
#include "options.h"

namespace pomai::storage {

/**
 * PulpView: Zero-copy read-only view of an Aril's approximate Pulp representation.
 */
class PulpView {
public:
    constexpr PulpView() = default;
    constexpr PulpView(const uint8_t* codes, uint32_t count, uint32_t dim,
                       float min_val, float inv_scale, uint8_t quant_type = 1)
        : codes_(codes), count_(count), dim_(dim),
          min_val_(min_val), inv_scale_(inv_scale), quant_type_(quant_type) {}

    [[nodiscard]] uint32_t count() const noexcept { return count_; }
    [[nodiscard]] uint32_t dim() const noexcept { return dim_; }
    [[nodiscard]] float quant_min() const noexcept { return min_val_; }
    [[nodiscard]] float quant_inv_scale() const noexcept { return inv_scale_; }
    [[nodiscard]] uint8_t quant_type() const noexcept { return quant_type_; }
    [[nodiscard]] const uint8_t* data() const noexcept { return codes_; }

    [[nodiscard]] std::span<const uint8_t> GetCodes(uint32_t slot) const noexcept {
        if (!codes_ || slot >= count_) return {};
        return {codes_ + static_cast<size_t>(slot) * dim_, dim_};
    }

    /**
     * Taste: Computes approximate distance against query using compressed Pulp.
     * Higher score = better candidate (negated L2 for Euclidean metric).
     */
    [[nodiscard]] float Taste(std::span<const float> query, uint32_t slot,
                              pomai::MetricType metric, float query_sum = 0.0f) const noexcept {
        if (!codes_ || slot >= count_) return -1e9f;
        auto codes = GetCodes(slot);

        const bool is_ip = (metric == pomai::MetricType::kInnerProduct ||
                            metric == pomai::MetricType::kCosine);
        if (is_ip) {
            return pomai::core::DotSq8(query, codes, min_val_, inv_scale_, query_sum);
        } else {
            const float max_val = min_val_ + 255.0f * inv_scale_;
            return -pomai::core::L2SqSq8(query, codes, min_val_, max_val);
        }
    }

private:
    const uint8_t* codes_{nullptr};
    uint32_t count_{0};
    uint32_t dim_{0};
    float min_val_{0.0f};
    float inv_scale_{0.0f};
    uint8_t quant_type_{1}; // 1 = SQ8
};

/**
 * PulpBuilder: Quantizes raw vectors into an aligned, contiguous Pulp block.
 */
class PulpBuilder {
public:
    PulpBuilder(uint32_t dim, uint8_t quant_type = 1);

    /**
     * Train: Determines global quantization bounds [min, max] across vectors.
     */
    void Train(const std::vector<std::span<const float>>& vectors);

    /**
     * Encode: Encodes a single FP32 vector into quantized Pulp codes.
     */
    void EncodeAppend(std::span<const float> vec);

    [[nodiscard]] const std::vector<uint8_t>& buffer() const noexcept { return buffer_; }
    [[nodiscard]] uint32_t count() const noexcept { return count_; }
    [[nodiscard]] uint32_t dim() const noexcept { return dim_; }
    [[nodiscard]] float min_val() const noexcept { return min_val_; }
    [[nodiscard]] float inv_scale() const noexcept { return inv_scale_; }
    [[nodiscard]] uint8_t quant_type() const noexcept { return quant_type_; }

private:
    uint32_t dim_{0};
    uint32_t count_{0};
    uint8_t quant_type_{1};
    float min_val_{0.0f};
    float scale_{0.0f};
    float inv_scale_{0.0f};
    std::vector<uint8_t> buffer_;
};

} // namespace pomai::storage
