// pomai/seed_kernel.h — Exact FP32 vector representation (Seed Kernel)
//
// In the Pomegranate Engine:
// - Seed Kernel contains the exact source FP32 vectors.
// - Physically separate from Pulp to avoid loading FP32 data during approximate search.
// - Accessed only during Stage 5 (Rerank) for candidate verification.
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
 * SeedKernelView: Zero-copy read-only view of the exact FP32 vectors in an Aril.
 */
class SeedKernelView {
public:
    constexpr SeedKernelView() = default;
    constexpr SeedKernelView(const float* data, uint32_t count, uint32_t dim)
        : data_(data), count_(count), dim_(dim) {}

    [[nodiscard]] uint32_t count() const noexcept { return count_; }
    [[nodiscard]] uint32_t dim() const noexcept { return dim_; }
    [[nodiscard]] const float* data() const noexcept { return data_; }
    [[nodiscard]] size_t size_bytes() const noexcept {
        return static_cast<size_t>(count_) * dim_ * sizeof(float);
    }

    [[nodiscard]] std::span<const float> GetVector(uint32_t slot) const noexcept {
        if (!data_ || slot >= count_) return {};
        return {data_ + static_cast<size_t>(slot) * dim_, dim_};
    }

    /**
     * Rerank: Computes exact FP32 distance for the candidate slot.
     * Higher score = better candidate (delegates to canonical ComputeMetricScore).
     */
    [[nodiscard]] float Rerank(std::span<const float> query, uint32_t slot,
                               pomai::MetricType metric) const noexcept {
        if (!data_ || slot >= count_) return -1e9f;
        auto vec = GetVector(slot);
        return pomai::core::ComputeMetricScore(metric, query, vec);
    }

private:
    const float* data_{nullptr};
    uint32_t count_{0};
    uint32_t dim_{0};
};

/**
 * SeedKernelBuilder: Assembles contiguous 64-byte aligned FP32 vectors for an Aril.
 */
class SeedKernelBuilder {
public:
    explicit SeedKernelBuilder(uint32_t dim) : dim_(dim), count_(0) {}

    void Append(std::span<const float> vec) {
        if (vec.size() != dim_) return;
        const size_t prev = data_.size();
        data_.resize(prev + dim_);
        std::copy(vec.begin(), vec.end(), data_.begin() + static_cast<std::ptrdiff_t>(prev));
        ++count_;
    }

    [[nodiscard]] const std::vector<float>& data() const noexcept { return data_; }
    [[nodiscard]] uint32_t count() const noexcept { return count_; }
    [[nodiscard]] uint32_t dim() const noexcept { return dim_; }
    [[nodiscard]] size_t size_bytes() const noexcept { return data_.size() * sizeof(float); }

private:
    uint32_t dim_{0};
    uint32_t count_{0};
    std::vector<float> data_;
};

} // namespace pomai::storage
