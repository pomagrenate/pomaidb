// pomai/rind.h — Rind: Single mutable ingestion & WAL boundary
//
// In the Pomegranate Engine:
// - Rind is the tough outer layer that absorbs writes and guarantees durability.
// - Encapsulates active MemTable, WAL, and frozen MemTables awaiting Pressing.
// - Supports RAM-first immediate visibility for queries.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <psync/psync.h>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "memtable.h"
#include "metadata.h"
#include "options.h"
#include "status.h"
#include "types.h"
#include "wal.h"
#include "utils/env.h"
#include "utils/palloc_smart_ptr.h"

namespace pomai::ingest {

struct RindHit {
    VectorId id;
    float distance;
};

/**
 * RindTombstoneSnapshot: Immutable point-in-time snapshot of active/frozen memtable tombstones.
 * Acquired in O(1) time (single shared_ptr copy), accessed lock-free in query loops.
 */
class RindTombstoneSnapshot {
public:
    RindTombstoneSnapshot() = default;
    explicit RindTombstoneSnapshot(std::shared_ptr<const std::unordered_set<VectorId>> tombstones)
        : tombstones_(std::move(tombstones)) {}

    [[nodiscard]] bool IsDeleted(VectorId id) const noexcept {
        if (!tombstones_ || tombstones_->empty()) return false;
        return tombstones_->find(id) != tombstones_->end();
    }

    [[nodiscard]] size_t size() const noexcept {
        return tombstones_ ? tombstones_->size() : 0;
    }

    [[nodiscard]] bool empty() const noexcept {
        return !tombstones_ || tombstones_->empty();
    }

private:
    std::shared_ptr<const std::unordered_set<VectorId>> tombstones_;
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
    [[nodiscard]] RindTombstoneSnapshot CaptureTombstoneSnapshot() const;

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
    [[nodiscard]] uint32_t dimension() const noexcept { return dim_; }

    void ForEachEntry(const std::function<void(VectorId, std::span<const float>, bool is_deleted, const Metadata*)>& fn) const;

private:
    Env* env_;
    std::string db_dir_;
    uint32_t dim_;
    MetricType metric_;
    FsyncPolicy fsync_;
    size_t memtable_size_bytes_;

    mutable psync::SharedMutex mu_;  // CRITICAL FIX: Convert to reader-writer lock for read scalability
    std::shared_ptr<table::MemTable> active_memtable_;
    std::vector<std::shared_ptr<table::MemTable>> frozen_memtables_;
    alloc::UniquePtr<storage::Wal> wal_;
    bool opened_{false};

    // Fast Point-in-time tombstone snapshot tracking
    std::unordered_set<VectorId> tombstones_;
    mutable psync::Mutex snapshot_mu_;
    mutable std::shared_ptr<const std::unordered_set<VectorId>> cached_tombstones_;
    mutable std::atomic<bool> tombstones_dirty_{true};
};

} // namespace pomai::ingest
