// pomai/rind.h — Rind: Single mutable ingestion & WAL boundary
//
// In the Pomegranate Engine:
// - Rind is the tough outer layer that absorbs writes and guarantees durability.
// - Encapsulates active MemTable, WAL, and frozen MemTables awaiting Pressing.
// - Supports RAM-first immediate visibility for queries.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "memtable.h"
#include "metadata.h"
#include "options.h"
#include "status.h"
#include "types.h"
#include "wal.h"
#include "utils/env.h"

namespace pomai::ingest {

struct RindHit {
    VectorId id;
    float distance;
};

class Rind {
public:
    Rind(Env* env,
         std::string db_dir,
         uint32_t dim,
         MetricType metric,
         FsyncPolicy fsync = FsyncPolicy::kNever,
         size_t memtable_size_bytes = 64 * 1024 * 1024);

    ~Rind();

    Status Open();
    Status Close();

    // Ingestion
    Status Put(VectorId id, std::span<const float> vec, const Metadata* meta = nullptr);
    Status PutBatch(std::span<const VectorId> ids, std::span<const float> vecs, uint32_t dim);
    Status Delete(VectorId id);

    // Point lookups
    Status Get(VectorId id, std::vector<float>* out, Metadata* meta = nullptr) const;
    [[nodiscard]] bool Contains(VectorId id) const;
    [[nodiscard]] bool IsDeleted(VectorId id) const;

    // Durability & Lifecycle
    Status Flush();
    Status Freeze();

    /**
     * TakeFrozenMemtables: Transference of frozen tables to the Press compaction engine.
     */
    std::vector<std::shared_ptr<table::MemTable>> TakeFrozenMemtables();

    /**
     * Taste: Quick linear/brute-force scan over active and frozen memtables.
     */
    void Taste(std::span<const float> query, uint32_t topk, MetricType metric,
               std::vector<RindHit>* out_hits) const;

    [[nodiscard]] size_t BytesUsed() const noexcept;
    [[nodiscard]] size_t ActiveCount() const noexcept;
    [[nodiscard]] bool HasFrozen() const noexcept;

    void ForEachEntry(const std::function<void(VectorId, std::span<const float>, bool is_deleted, const Metadata*)>& fn) const;

private:
    Env* env_;
    std::string db_dir_;
    uint32_t dim_;
    MetricType metric_;
    FsyncPolicy fsync_;
    size_t memtable_size_bytes_;

    mutable std::mutex mu_;
    std::shared_ptr<table::MemTable> active_memtable_;
    std::vector<std::shared_ptr<table::MemTable>> frozen_memtables_;
    std::unique_ptr<storage::Wal> wal_;
    bool opened_{false};
};

} // namespace pomai::ingest
