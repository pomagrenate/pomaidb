// hnsw_index.h — PomaiDB wrapper around faiss::IndexHNSWFlat.
//
// Phase 3: Wraps FAISS HNSW for use as a drop-in sidecar index per runtime.
// Activated via DBOptions::index_type = IndexType::kHNSW.
// Backward-compatible: when not set, IvfFlatIndex is used as before.
//
// Phase 4: Save/Load via faiss::index_io for .idx sidecar persistence.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include "util/aligned_vector.h"

#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/options.h"

namespace pomai::hnsw {
class HNSW;
}

namespace pomai::index {

/// Options for the HNSW index.
struct HnswOptions {
    int M            = 32;    // Number of HNSW neighbors per node. Higher = better recall, more RAM.
    int ef_construction = 200; // Candidates during build. Higher = better graph quality.
    int ef_search    = 64;    // Candidates during query. Tunable at query time.
};

/// Per-runtime HNSW index. Thread-safe for concurrent reads; single-writer Add().
class HnswIndex {
public:
    /// Create an empty HNSW index (not yet trained/populated).
    HnswIndex(uint32_t dim, HnswOptions opts = {}, pomai::MetricType metric = pomai::MetricType::kInnerProduct);
    ~HnswIndex();

    // Non-copyable
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;

    // ── Build Phase ───────────────────────────────────────────────────────────
    /// Add one vector with the given PomaiDB VectorId.
    /// NOTE: FAISS HNSW stores vectors contiguously; internal FAISS ids are
    ///       sequential. We keep a mapping faiss_id → VectorId.
    pomai::Status Add(VectorId id, std::span<const float> vec);

    /// Add a batch of vectors.
    pomai::Status AddBatch(const VectorId* ids,
                           const float*    vecs,
                           std::size_t     n);

    // ── Query Phase ───────────────────────────────────────────────────────────
    /// Approximate nearest neighbor search.
    /// @param query     Query vector (dim() floats).
    /// @param topk      Number of results requested.
    /// @param ef_search Override for ef_search (0 = use default from HnswOptions).
    /// @param out_ids   Output VectorIds (size topk).
    /// @param out_dists Output distances (size topk).
    pomai::Status Search(std::span<const float> query,
                         uint32_t               topk,
                         int                    ef_search,
                         std::vector<VectorId>* out_ids,
                         std::vector<float>*    out_dists) const;

    // ── Metadata ──────────────────────────────────────────────────────────────
    uint32_t    dim()   const { return dim_; }
    std::size_t count() const { return id_map_.size(); }
    HnswOptions opts()  const { return opts_; }

    // ── Persistence (Phase 4) ─────────────────────────────────────────────────
    /// Write index to path with full vector pool. Also writes entry_index_map
    /// in new v1 format (magic header).
    pomai::Status Save(const std::string& path) const;

    /// Write index without vector pool (v2 format). Requires SetEntryIndexMap()
    /// to have been called. The segment provides vectors at query time via
    /// SetVectorGetter().
    pomai::Status SaveNoPool(const std::string& path) const;

    /// Load from path. Handles old format (no magic), v1 (pool), v2 (no-pool).
    static pomai::Status Load(const std::string& path,
                              std::unique_ptr<HnswIndex>* out);

    // ── No-pool mode (graph-only RAM saving) ──────────────────────────────────
    /// Maps HNSW internal index → segment entry index (built at BuildIndex time).
    /// Must be set before SaveNoPool().
    void SetEntryIndexMap(std::vector<uint32_t> map) {
        entry_index_map_ = std::move(map);
    }

    /// Inject a vector getter for no-pool mode. Called by SegmentReader after
    /// Load() when IsNoVectorPool() is true.
    /// @param getter  fn(entry_index) → const float*  (points into mmap).
    void SetVectorGetter(std::function<const float*(uint32_t)> getter) {
        vector_getter_ = std::move(getter);
    }

    /// True when loaded from a no-pool file (vector_pool_ is empty).
    bool IsNoVectorPool() const { return no_vector_pool_; }

private:
    uint32_t    dim_;
    HnswOptions opts_;

    // Owned native HNSW index
    std::unique_ptr<pomai::hnsw::HNSW> index_;

    // Vector storage for distance computation during Add/Search.
    // Empty when no_vector_pool_ == true (search uses vector_getter_ instead).
    pomai::util::AlignedVector<float> vector_pool_;
    pomai::MetricType metric_;

    // HNSW internal idx → PomaiDB VectorId mapping
    std::vector<VectorId> id_map_;

    // No-pool mode: HNSW internal idx → segment entry index
    std::vector<uint32_t> entry_index_map_;

    // When true, vector_pool_ is empty and vector_getter_ provides vectors.
    bool no_vector_pool_{false};

    // Segment mmap vector resolver (set by SegmentReader in no-pool mode).
    std::function<const float*(uint32_t)> vector_getter_;

    // File format magic for versioned header detection.
    static constexpr uint32_t kFileMagic = 0x504D4831u; // "PMH1"
};

} // namespace pomai::index
