// pomai/seed_kernel.h — Exact FP32 vector representation (Seed Kernel)
//
// In the Pomegranate Engine:
// - Seed Kernel contains the exact source FP32 vectors.
// - Physically separate from Pulp to avoid loading FP32 data during approximate search.
// - Accessed only during Stage 5 (Rerank) for candidate verification.
//
// Memory Layout (Bifurcated):
// - Vector Block: Pure floats, 64-byte aligned, NO metadata
// - Metadata Block: IDs, flags, separate buffer
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
 * VectorMetadata: Metadata separated from vector data
 * Kept in separate buffer to avoid cache pollution during SIMD scans
 */
struct alignas(64) VectorMetadata {
    uint64_t id{0};
    uint32_t flags{0};
    uint32_t reserved{0};
};

/**
 * SeedKernelView: Zero-copy read-only view of the exact FP32 vectors in an Aril.
 * Vectors are stored in pure float array (no metadata interleaved)
 */
class SeedKernelView {
public:
    constexpr SeedKernelView() = default;
    constexpr SeedKernelView(const float* data, uint32_t count, uint32_t dim,
                           const VectorMetadata* metadata = nullptr)
        : data_(data), metadata_(metadata), count_(count), dim_(dim) {}

    [[nodiscard]] uint32_t count() const noexcept { return count_; }
    [[nodiscard]] uint32_t dim() const noexcept { return dim_; }
    [[nodiscard]] const float* data() const noexcept { return data_; }
    [[nodiscard]] const VectorMetadata* metadata() const noexcept { return metadata_; }
    [[nodiscard]] size_t size_bytes() const noexcept {
        return static_cast<size_t>(count_) * dim_ * sizeof(float);
    }

    [[nodiscard]] std::span<const float> GetVector(uint32_t slot) const noexcept {
        if (!data_ || slot >= count_) return {};
        return {data_ + static_cast<size_t>(slot) * dim_, dim_};
    }

    /**
     * GetMetadata: Access metadata AFTER SIMD scan completes
     * Only called when we need ID/flags for Top-K results
     */
    [[nodiscard]] const VectorMetadata* GetMetadata(uint32_t slot) const noexcept {
        if (!metadata_ || slot >= count_) return nullptr;
        return metadata_ + slot;
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
    const VectorMetadata* metadata_{nullptr};
    uint32_t count_{0};
    uint32_t dim_{0};
};

/**
 * SeedKernelBuilder: Assembles contiguous 64-byte aligned FP32 vectors for an Aril.
 * Separates vector data from metadata to prevent cache pollution
 */
class SeedKernelBuilder {
public:
    explicit SeedKernelBuilder(uint32_t dim) : dim_(dim), count_(0) {}

    void Append(std::span<const float> vec, uint64_t id, uint32_t flags = 0) {
        if (vec.size() != dim_) return;
        
        // Append vector data (pure floats, no metadata)
        const size_t prev = data_.size();
        data_.resize(prev + dim_);
        std::copy(vec.begin(), vec.end(), data_.begin() + static_cast<std::ptrdiff_t>(prev));
        
        // Append metadata separately
        VectorMetadata meta;
        meta.id = id;
        meta.flags = flags;
        metadata_.push_back(meta);
        
        ++count_;
    }

    [[nodiscard]] const std::vector<float>& data() const noexcept { return data_; }
    [[nodiscard]] const std::vector<VectorMetadata>& metadata() const noexcept { return metadata_; }
    [[nodiscard]] uint32_t count() const noexcept { return count_; }
    [[nodiscard]] uint32_t dim() const noexcept { return dim_; }
    [[nodiscard]] size_t size_bytes() const noexcept { return data_.size() * sizeof(float); }
    [[nodiscard]] size_t metadata_bytes() const noexcept { return metadata_.size() * sizeof(VectorMetadata); }

private:
    uint32_t dim_{0};
    uint32_t count_{0};
    std::vector<float> data_;           // Pure vector data
    std::vector<VectorMetadata> metadata_;  // Separate metadata buffer
};

} // namespace pomai::storage
