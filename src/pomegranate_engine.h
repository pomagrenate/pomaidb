// pomai/pomegranate_engine.h — Unified Pomegranate Architecture Vector Engine
//
// In the Pomegranate Engine:
// - Rind: Single mutable ingestion boundary & WAL buffer
// - Compass: Spatial routing & geometric peel pruning
// - Locules & Arils: Immutable container segments with Pulp (SQ8) & Seed Kernel (FP32)
// - Press: Drying, compaction, reseeding
// - FruitMap: Atomic manifest & generation tracking
// - PomegranateQuery: 5-stage search pipeline
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <psync/psync.h>

#include "compass.h"
#include "fruit_map.h"
#include "iterator.h"
#include "locule.h"
#include "metadata.h"
#include "options.h"
#include "pomegranate_query.h"
#include "press.h"
#include "rind.h"
#include "search.h"
#include "snapshot.h"
#include "status.h"
#include "types.h"
#include "utils/env.h"
#include <ptask/ptask.h>

namespace pomai::core {

class PomegranateEngine {
public:
    PomegranateEngine(DBOptions opt,
                      MetricType metric = MetricType::kL2,
                      Env* env = nullptr);
    ~PomegranateEngine();

    PomegranateEngine(const PomegranateEngine&) = delete;
    PomegranateEngine& operator=(const PomegranateEngine&) = delete;

    Status Open();
    Status Close();

    // Ingestion
    Status Put(VectorId id, std::span<const float> vec);
    Status Put(VectorId id, std::span<const float> vec, const Metadata& meta);
    Status PutBatch(const std::vector<VectorId>& ids,
                    const std::vector<std::span<const float>>& vectors);
    Status PutBatch(const std::vector<VectorId>& ids,
                    const std::vector<std::vector<float>>& vectors);
    Status PutBatch(std::span<const VectorId> ids,
                    std::span<const float> vectors,
                    std::size_t dimension);

    // Lookups
    Status Get(VectorId id, std::vector<float>* out);
    Status Get(VectorId id, std::vector<float>* out, Metadata* out_meta);
    Status Exists(VectorId id, bool* exists);
    Status Delete(VectorId id);

    // Maintenance
    Status Flush();
    Status Freeze();
    Status Compact();

    // Search
    Status Search(std::span<const float> query,
                  uint32_t topk,
                  SearchResult* out);
    Status Search(std::span<const float> query,
                  uint32_t topk,
                  const SearchOptions& opts,
                  SearchResult* out);
    Status Search(std::span<const float> query,
                  uint32_t topk,
                  const SearchOptions& opts,
                  SearchHitSink& sink);
    Status SearchBatch(std::span<const float> queries,
                       uint32_t num_queries,
                       uint32_t topk,
                       const SearchOptions& opts,
                       std::vector<SearchResult>* out);

    // Snapshot & Iteration
    Status GetSnapshot(std::shared_ptr<Snapshot>* out);
    Status NewIterator(std::unique_ptr<SnapshotIterator>* out);
    Status NewIterator(const std::shared_ptr<Snapshot>& snap,
                       std::unique_ptr<SnapshotIterator>* out);

    [[nodiscard]] size_t MemTableBytesUsed() const noexcept;
    [[nodiscard]] const DBOptions& options() const noexcept { return opt_; }
    [[nodiscard]] MetricType metric() const noexcept { return metric_; }

private:
    DBOptions opt_;
    MetricType metric_;
    Env* env_;

    std::unique_ptr<ingest::Rind> rind_;
    std::unique_ptr<manifest::FruitMap> fruit_map_;
    std::unique_ptr<compact::Press> press_;
    std::unique_ptr<ptask::ThreadPool> thread_pool_;
    mutable psync::Mutex compact_mu_;
    bool opened_{false};
};

} // namespace pomai::core
