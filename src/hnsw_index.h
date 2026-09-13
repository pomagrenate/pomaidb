// src/hnsw_index.h — PomaiDB production wrapper around upstream nmslib/hnswlib
//
// Algorithmic Authority: nmslib/hnswlib (upstream commit 058d7a866e462c00f0a6dea8660969379d5916bd)
// Integrated via PomaiDistanceSpace with PomaiDB SIMD distance kernels.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "options.h"
#include "status.h"
#include "types.h"

namespace pomai::index {

/// Production HNSW configuration parameters
struct HnswOptions {
    size_t M = 16;
    size_t ef_construction = 200;
    size_t ef_search = 64;
    size_t initial_max_elements = 1024;
    size_t random_seed = 100;
};

/// Interface for in-graph filtering of VectorIds during search
class IdFilter {
public:
    virtual ~IdFilter() = default;
    virtual bool IsAllowed(VectorId id) = 0;
};

/// High-performance production HNSW index wrapping hnswlib::HierarchicalNSW<float>
class HnswIndex {
public:
    HnswIndex(uint32_t dim, HnswOptions opts = {}, pomai::MetricType metric = pomai::MetricType::kL2);
    ~HnswIndex();

    // Non-copyable, movable
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;
    HnswIndex(HnswIndex&&) noexcept;
    HnswIndex& operator=(HnswIndex&&) noexcept;

    // ── Build Phase ───────────────────────────────────────────────────────────
    pomai::Status Add(VectorId id, std::span<const float> vec);
    pomai::Status AddBatch(const VectorId* ids, const float* vecs, std::size_t n);

    // ── Query Phase ───────────────────────────────────────────────────────────
    /// Approximate nearest neighbor search.
    /// @param query     Query vector (dim() floats).
    /// @param topk      Number of results requested.
    /// @param ef_search Override for ef_search (0 = use default from HnswOptions; enforces ef_search >= topk).
    /// @param out_ids   Output VectorIds (sorted closest first).
    /// @param out_dists Output distances (sorted closest first).
    /// @param filter    Optional in-graph filter for tombstones/metadata.
    pomai::Status Search(std::span<const float> query,
                         uint32_t topk,
                         int ef_search,
                         std::vector<VectorId>* out_ids,
                         std::vector<float>* out_dists,
                         IdFilter* filter = nullptr) const;

    // ── Metadata ──────────────────────────────────────────────────────────────
    [[nodiscard]] uint32_t dim() const noexcept { return dim_; }
    [[nodiscard]] std::size_t count() const;
    [[nodiscard]] HnswOptions opts() const noexcept { return opts_; }
    [[nodiscard]] pomai::MetricType metric() const noexcept { return metric_; }

    // ── Persistence ───────────────────────────────────────────────────────────
    pomai::Status SaveToStream(std::ostream& out) const;
    pomai::Status LoadFromStream(std::istream& in);

    pomai::Status SaveToBuffer(std::vector<uint8_t>* out) const;
    pomai::Status LoadFromBuffer(const uint8_t* data, size_t len);

    pomai::Status Save(const std::string& path) const;
    static pomai::Status Load(const std::string& path,
                              std::unique_ptr<HnswIndex>* out);
    static pomai::Status Load(const std::string& path,
                              uint32_t dim,
                              pomai::MetricType metric,
                              std::unique_ptr<HnswIndex>* out);

    // ── Legacy Sidecar Compatibility ──────────────────────────────────────────
    void SetEntryIndexMap(std::vector<uint32_t> map) { entry_index_map_ = std::move(map); }
    void SetVectorGetter(std::function<const float*(uint32_t)> getter) { vector_getter_ = std::move(getter); }
    [[nodiscard]] bool IsNoVectorPool() const noexcept { return no_vector_pool_; }
    pomai::Status SaveNoPool(const std::string& path) const { return Save(path); }

private:
    uint32_t dim_;
    HnswOptions opts_;
    pomai::MetricType metric_;

    std::vector<uint32_t> entry_index_map_;
    std::function<const float*(uint32_t)> vector_getter_;
    bool no_vector_pool_{false};

    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pomai::index
