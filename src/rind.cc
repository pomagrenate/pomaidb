// pomai/rind.cc — Rind: Ingestion & WAL boundary implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "rind.h"

#include <algorithm>
#include <mutex>

#include "distance.h"

namespace pomai::ingest {

Rind::Rind(Env* env,
           std::string db_dir,
           uint32_t dim,
           MetricType metric,
           FsyncPolicy fsync,
           size_t memtable_size_bytes)
    : env_(env ? env : Env::Default()),
      db_dir_(std::move(db_dir)),
      dim_(dim),
      metric_(metric),
      fsync_(fsync),
      memtable_size_bytes_(memtable_size_bytes) {}

Rind::~Rind() {
    (void)Close();
}

Status Rind::Open() {
    std::unique_lock<std::shared_mutex> lock(mu_);
    if (opened_) return Status::Ok();

    Status s = env_->CreateDirIfMissing(db_dir_);
    if (!s.ok()) return s;

    active_memtable_ = std::make_shared<table::MemTable>(dim_, 4 * 1024 * 1024);

    wal_ = std::make_unique<storage::Wal>(env_, db_dir_, 0, 64 * 1024 * 1024, fsync_);
    s = wal_->Open();
    if (!s.ok()) return s;

    s = wal_->ReplayInto(*active_memtable_);
    if (!s.ok()) return s;

    // Populate initial tombstones from replayed active memtable
    tombstones_.clear();
    auto cursor = active_memtable_->CreateCursor();
    table::MemTable::CursorEntry entry;
    while (cursor.Next(&entry)) {
        if (entry.is_deleted) {
            tombstones_.insert(entry.id);
        }
    }
    tombstones_dirty_ = true;

    opened_ = true;
    return Status::Ok();
}

Status Rind::Close() {
    std::unique_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return Status::Ok();

    if (wal_) {
        (void)wal_->Flush();
    }
    opened_ = false;
    return Status::Ok();
}

Status Rind::Put(VectorId id, std::span<const float> vec, const Metadata* meta) {
    if (vec.size() != dim_) {
        return Status::InvalidArgument("dimension mismatch");
    }
    for (float v : vec) {
        if (!std::isfinite(v)) {
            return Status::InvalidArgument("vector contains non-finite values (NaN or Inf)");
        }
    }

    std::unique_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return Status::Corruption("rind is not open");

    Status s;
    if (wal_) {
        if (meta) {
            s = wal_->AppendPut(id, vec, *meta);
        } else {
            s = wal_->AppendPut(id, vec);
        }
        if (!s.ok()) return s;
    }

    if (meta) {
        s = active_memtable_->Put(id, vec, *meta);
    } else {
        s = active_memtable_->Put(id, vec);
    }
    if (!s.ok()) return s;

    tombstones_.erase(id);
    tombstones_dirty_ = true;

    if (active_memtable_->BytesUsed() >= memtable_size_bytes_) {
        // Auto-freeze when threshold exceeded
        (void)wal_->Flush();
        frozen_memtables_.push_back(active_memtable_);
        active_memtable_ = std::make_shared<table::MemTable>(dim_, 4 * 1024 * 1024);
    }

    return Status::Ok();
}

Status Rind::PutBatch(std::span<const VectorId> ids, std::span<const float> vecs, uint32_t dim) {
    if (dim != dim_ || vecs.size() != ids.size() * dim) {
        return Status::InvalidArgument("invalid batch dimension or vector size");
    }
    for (float v : vecs) {
        if (!std::isfinite(v)) {
            return Status::InvalidArgument("vector contains non-finite values (NaN or Inf)");
        }
    }

    std::unique_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return Status::Corruption("rind is not open");

    std::vector<VectorId> ids_vec(ids.begin(), ids.end());
    std::vector<VectorView> views;
    views.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        views.emplace_back(vecs.data() + i * dim, dim);
    }

    Status s;
    if (wal_) {
        s = wal_->AppendBatch(ids_vec, views);
        if (!s.ok()) return s;
    }

    s = active_memtable_->PutBatch(ids_vec, views);
    if (!s.ok()) return s;

    for (VectorId id : ids) {
        tombstones_.erase(id);
    }
    tombstones_dirty_ = true;

    if (active_memtable_->BytesUsed() >= memtable_size_bytes_) {
        (void)wal_->Flush();
        frozen_memtables_.push_back(active_memtable_);
        active_memtable_ = std::make_shared<table::MemTable>(dim_, 4 * 1024 * 1024);
    }

    return Status::Ok();
}

Status Rind::Delete(VectorId id) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return Status::Corruption("rind is not open");

    if (wal_) {
        Status s = wal_->AppendDelete(id);
        if (!s.ok()) return s;
    }

    Status s = active_memtable_->Delete(id);
    if (s.ok()) {
        tombstones_.insert(id);
        tombstones_dirty_ = true;
    }
    return s;
}

Status Rind::Get(VectorId id, std::vector<float>* out, Metadata* meta) const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return Status::Corruption("rind is not open");

    // Check active memtable
    if (active_memtable_->IsTombstone(id)) {
        return Status::NotFound("vector is tombstoned in active memtable");
    }
    const float* ptr = nullptr;
    Status s;
    if (meta) {
        s = active_memtable_->Get(id, &ptr, meta);
    } else {
        s = active_memtable_->Get(id, &ptr);
    }
    if (s.ok() && ptr) {
        if (out) out->assign(ptr, ptr + dim_);
        return Status::Ok();
    }

    // Check frozen memtables in reverse
    for (auto it = frozen_memtables_.rbegin(); it != frozen_memtables_.rend(); ++it) {
        const auto& table = *it;
        if (!table) continue;

        if (table->IsTombstone(id)) {
            return Status::NotFound("vector is tombstoned in frozen memtable");
        }

        const float* fptr = nullptr;
        if (meta) {
            s = table->Get(id, &fptr, meta);
        } else {
            s = table->Get(id, &fptr);
        }
        if (s.ok() && fptr) {
            if (out) out->assign(fptr, fptr + dim_);
            return Status::Ok();
        }
    }

    return Status::NotFound("vector not found in rind");
}

bool Rind::Contains(VectorId id) const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return false;

    if (active_memtable_->IsTombstone(id)) return false;
    if (active_memtable_->GetPtr(id) != nullptr) return true;

    for (auto it = frozen_memtables_.rbegin(); it != frozen_memtables_.rend(); ++it) {
        const auto& table = *it;
        if (!table) continue;
        if (table->IsTombstone(id)) return false;
        if (table->GetPtr(id) != nullptr) return true;
    }
    return false;
}

bool Rind::IsDeleted(VectorId id) const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return false;
    return tombstones_.find(id) != tombstones_.end();
}

RindTombstoneSnapshot Rind::CaptureTombstoneSnapshot() const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return RindTombstoneSnapshot{};

    if (tombstones_dirty_.load(std::memory_order_acquire) || !cached_tombstones_) {
        std::lock_guard<std::mutex> snap_lock(snapshot_mu_);
        if (tombstones_dirty_.load(std::memory_order_relaxed) || !cached_tombstones_) {
            cached_tombstones_ = std::make_shared<const std::unordered_set<VectorId>>(tombstones_);
            tombstones_dirty_.store(false, std::memory_order_release);
        }
    }
    return RindTombstoneSnapshot(cached_tombstones_);
}

Status Rind::Flush() {
    std::unique_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return Status::Corruption("rind is not open");
    if (wal_) return wal_->Flush();
    return Status::Ok();
}

Status Rind::Freeze() {
    std::unique_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return Status::Corruption("rind is not open");
    if (active_memtable_->GetCount() == 0) return Status::Ok();

    if (wal_) {
        Status s = wal_->Flush();
        if (!s.ok()) return s;
    }

    frozen_memtables_.push_back(active_memtable_);
    active_memtable_ = std::make_shared<table::MemTable>(dim_, 4 * 1024 * 1024);
    return Status::Ok();
}

std::vector<std::shared_ptr<table::MemTable>> Rind::TakeFrozenMemtables() {
    std::unique_lock<std::shared_mutex> lock(mu_);
    auto res = std::move(frozen_memtables_);
    frozen_memtables_.clear();

    // Re-sync tombstones from active_memtable_
    tombstones_.clear();
    if (active_memtable_) {
        auto cursor = active_memtable_->CreateCursor();
        table::MemTable::CursorEntry entry;
        while (cursor.Next(&entry)) {
            if (entry.is_deleted) {
                tombstones_.insert(entry.id);
            }
        }
    }
    tombstones_dirty_ = true;
    return res;
}

void Rind::Taste(std::span<const float> query, uint32_t topk, MetricType metric,
                 std::vector<RindHit>* out_hits) const {
    (void)topk;
    if (!out_hits) return;

    // CRITICAL FIX: Snapshot memtable pointers under minimal lock, then compute SIMD distances outside lock
    std::shared_ptr<table::MemTable> active;
    std::vector<std::shared_ptr<table::MemTable>> frozen;
    uint32_t dim_snapshot;
    bool opened_snapshot;
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        if (!opened_) return;
        active = active_memtable_;
        frozen = frozen_memtables_;
        dim_snapshot = dim_;
        opened_snapshot = opened_;
    }

    if (!opened_snapshot) return;

    // Helper lambda to scan a memtable (executed entirely outside lock)
    auto scan_table = [&](const table::MemTable& table) {
        auto cursor = table.CreateCursor();
        table::MemTable::CursorEntry entry;
        while (cursor.Next(&entry)) {
            if (entry.is_deleted || entry.vec.size() != dim_snapshot) continue;

            float score = core::ComputeMetricScore(metric, query, entry.vec);
            out_hits->push_back({entry.id, score});
        }
    };

    // Scan frozen memtables first, then active (all outside lock)
    for (const auto& table : frozen) {
        if (table) scan_table(*table);
    }
    if (active) {
        scan_table(*active);
    }
}

size_t Rind::BytesUsed() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mu_);
    size_t total = active_memtable_ ? active_memtable_->BytesUsed() : 0;
    for (const auto& t : frozen_memtables_) {
        if (t) total += t->BytesUsed();
    }
    return total;
}

size_t Rind::ActiveCount() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mu_);
    return active_memtable_ ? active_memtable_->GetCount() : 0;
}

bool Rind::HasFrozen() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mu_);
    return !frozen_memtables_.empty();
}

void Rind::ForEachEntry(const std::function<void(VectorId, std::span<const float>, bool is_deleted, const Metadata*)>& fn) const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    if (!opened_) return;

    for (const auto& table : frozen_memtables_) {
        if (!table) continue;
        auto cursor = table->CreateCursor();
        table::MemTable::CursorEntry entry;
        while (cursor.Next(&entry)) {
            fn(entry.id, entry.vec, entry.is_deleted, entry.meta);
        }
    }

    if (active_memtable_) {
        auto cursor = active_memtable_->CreateCursor();
        table::MemTable::CursorEntry entry;
        while (cursor.Next(&entry)) {
            fn(entry.id, entry.vec, entry.is_deleted, entry.meta);
        }
    }
}

} // namespace pomai::ingest
