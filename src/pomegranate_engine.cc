// pomai/pomegranate_engine.cc — Unified Pomegranate Architecture Vector Engine implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "pomegranate_engine.h"

#include <algorithm>
#include <map>

namespace pomai::core {

namespace {

class PomegranateSnapshotImpl : public Snapshot {
public:
    struct Item {
        VectorId id{0};
        std::vector<float> vec;
        Metadata meta;
    };

    explicit PomegranateSnapshotImpl(std::vector<Item> items)
        : items_(std::move(items)) {}

    [[nodiscard]] const std::vector<Item>& items() const noexcept { return items_; }

private:
    std::vector<Item> items_;
};

class PomegranateSnapshotIteratorImpl : public SnapshotIterator {
public:
    explicit PomegranateSnapshotIteratorImpl(std::shared_ptr<const PomegranateSnapshotImpl> snap)
        : snap_(std::move(snap)), idx_(0) {}

    bool Next() override {
        if (!snap_ || idx_ >= snap_->items().size()) {
            return false;
        }
        ++idx_;
        return Valid();
    }

    [[nodiscard]] VectorId id() const override {
        if (!Valid()) return 0;
        return snap_->items()[idx_].id;
    }

    [[nodiscard]] std::span<const float> vector() const override {
        if (!Valid()) return {};
        const auto& v = snap_->items()[idx_].vec;
        return {v.data(), v.size()};
    }

    [[nodiscard]] const Metadata* metadata() const override {
        if (!Valid()) return nullptr;
        return &snap_->items()[idx_].meta;
    }

    [[nodiscard]] bool Valid() const override {
        return snap_ && idx_ < snap_->items().size();
    }

private:
    std::shared_ptr<const PomegranateSnapshotImpl> snap_;
    size_t idx_{0};
};

} // namespace

PomegranateEngine::PomegranateEngine(DBOptions opt, MetricType metric, Env* env)
    : opt_(std::move(opt)),
      metric_(metric),
      env_(env ? env : (opt_.env ? opt_.env : Env::Default())) {}

PomegranateEngine::~PomegranateEngine() {
    (void)Close();
}

Status PomegranateEngine::Open() {
    if (opened_) return Status::Ok();

    fruit_map_ = std::make_unique<manifest::FruitMap>(env_, opt_.path, opt_.dim, metric_);
    Status s = fruit_map_->Open();
    if (!s.ok()) return s;

    size_t mem_threshold = (opt_.memtable_flush_threshold_mb > 0)
                               ? static_cast<size_t>(opt_.memtable_flush_threshold_mb) * 1024 * 1024
                               : 64 * 1024 * 1024;
    rind_ = std::make_unique<ingest::Rind>(env_, opt_.path, opt_.dim, metric_, opt_.fsync, mem_threshold);
    s = rind_->Open();
    if (!s.ok()) return s;

    compact::PressOptions press_opts;
    press_opts.index_params = opt_.index_params;
    press_ = std::make_unique<compact::Press>(env_, opt_.path, opt_.dim, metric_, press_opts);

    opened_ = true;
    return Status::Ok();
}

Status PomegranateEngine::Close() {
    if (!opened_) return Status::Ok();

    if (rind_) {
        (void)rind_->Flush();
        (void)rind_->Close();
    }
    opened_ = false;
    return Status::Ok();
}

Status PomegranateEngine::Put(VectorId id, std::span<const float> vec) {
    if (!opened_) return Status::Corruption("engine not open");
    return rind_->Put(id, vec, nullptr);
}

Status PomegranateEngine::Put(VectorId id, std::span<const float> vec, const Metadata& meta) {
    if (!opened_) return Status::Corruption("engine not open");
    return rind_->Put(id, vec, &meta);
}

Status PomegranateEngine::PutBatch(const std::vector<VectorId>& ids,
                                  const std::vector<std::span<const float>>& vectors) {
    if (!opened_) return Status::Corruption("engine not open");
    if (ids.size() != vectors.size()) return Status::InvalidArgument("size mismatch");

    for (size_t i = 0; i < ids.size(); ++i) {
        Status s = rind_->Put(ids[i], vectors[i], nullptr);
        if (!s.ok()) return s;
    }
    return Status::Ok();
}

Status PomegranateEngine::PutBatch(const std::vector<VectorId>& ids,
                                  const std::vector<std::vector<float>>& vectors) {
    if (!opened_) return Status::Corruption("engine not open");
    if (ids.size() != vectors.size()) return Status::InvalidArgument("size mismatch");

    for (size_t i = 0; i < ids.size(); ++i) {
        Status s = rind_->Put(ids[i], vectors[i], nullptr);
        if (!s.ok()) return s;
    }
    return Status::Ok();
}

Status PomegranateEngine::PutBatch(std::span<const VectorId> ids,
                                  std::span<const float> vectors,
                                  std::size_t dimension) {
    if (!opened_) return Status::Corruption("engine not open");
    return rind_->PutBatch(ids, vectors, static_cast<uint32_t>(dimension));
}

Status PomegranateEngine::Get(VectorId id, std::vector<float>* out) {
    return Get(id, out, nullptr);
}

Status PomegranateEngine::Get(VectorId id, std::vector<float>* out, Metadata* out_meta) {
    if (!opened_) return Status::Corruption("engine not open");

    if (rind_->IsDeleted(id)) {
        return Status::NotFound("vector is tombstoned");
    }

    Status s = rind_->Get(id, out, out_meta);
    if (s.ok()) return Status::Ok();

    auto snap = fruit_map_->CurrentSnapshot();
    if (snap) {
        return snap->Get(id, out, out_meta);
    }

    return Status::NotFound("vector not found");
}

Status PomegranateEngine::Exists(VectorId id, bool* exists) {
    if (!opened_) return Status::Corruption("engine not open");
    if (!exists) return Status::InvalidArgument("null exists pointer");

    if (rind_->IsDeleted(id)) {
        *exists = false;
        return Status::Ok();
    }
    if (rind_->Contains(id)) {
        *exists = true;
        return Status::Ok();
    }

    auto snap = fruit_map_->CurrentSnapshot();
    if (snap) {
        *exists = snap->Contains(id);
        return Status::Ok();
    }

    *exists = false;
    return Status::Ok();
}

Status PomegranateEngine::Delete(VectorId id) {
    if (!opened_) return Status::Corruption("engine not open");
    return rind_->Delete(id);
}

Status PomegranateEngine::Flush() {
    if (!opened_) return Status::Corruption("engine not open");
    return rind_->Flush();
}

Status PomegranateEngine::Freeze() {
    if (!opened_) return Status::Corruption("engine not open");
    Status s = rind_->Freeze();
    if (!s.ok()) return s;

    if (rind_->HasFrozen()) {
        s = press_->Compact(rind_.get(), fruit_map_.get());
        if (!s.ok()) return s;
    }
    return Status::Ok();
}

Status PomegranateEngine::Compact() {
    if (!opened_) return Status::Corruption("engine not open");
    return press_->Compact(rind_.get(), fruit_map_.get());
}

Status PomegranateEngine::Search(std::span<const float> query,
                                uint32_t topk,
                                SearchResult* out) {
    return Search(query, topk, {}, out);
}

Status PomegranateEngine::Search(std::span<const float> query,
                                uint32_t topk,
                                const SearchOptions& opts,
                                SearchResult* out) {
    if (!opened_) return Status::Corruption("engine not open");
    auto snap = fruit_map_->CurrentSnapshot();
    return query::PomegranateQuery::Execute(query, topk, opts, metric_, snap.get(), rind_.get(), out);
}

Status PomegranateEngine::Search(std::span<const float> query,
                                uint32_t topk,
                                const SearchOptions& opts,
                                SearchHitSink& sink) {
    if (!opened_) return Status::Corruption("engine not open");
    auto snap = fruit_map_->CurrentSnapshot();
    return query::PomegranateQuery::Execute(query, topk, opts, metric_, snap.get(), rind_.get(), sink);
}

Status PomegranateEngine::SearchBatch(std::span<const float> queries,
                                     uint32_t num_queries,
                                     uint32_t topk,
                                     const SearchOptions& opts,
                                     std::vector<SearchResult>* out) {
    if (!opened_) return Status::Corruption("engine not open");
    if (!out) return Status::InvalidArgument("null out vector");

    uint32_t dim = opt_.dim;
    if (queries.size() != num_queries * dim) {
        return Status::InvalidArgument("queries size does not match num_queries * dim");
    }

    out->resize(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
        std::span<const float> query(queries.data() + q * dim, dim);
        Status s = Search(query, topk, opts, &(*out)[q]);
        if (!s.ok()) return s;
    }
    return Status::Ok();
}

Status PomegranateEngine::GetSnapshot(std::shared_ptr<Snapshot>* out) {
    if (!opened_) return Status::Corruption("engine not open");
    if (!out) return Status::InvalidArgument("null snapshot pointer");

    struct LiveItem {
        std::vector<float> vec;
        Metadata meta;
    };
    std::map<VectorId, LiveItem> live_items;

    auto snap = fruit_map_->CurrentSnapshot();
    if (snap) {
        for (const auto& loc : snap->locules()) {
            if (!loc) continue;
            for (const auto& aril : loc->arils()) {
                if (!aril) continue;
                for (const auto& entry : aril->directory()) {
                    if (aril->scar().IsDeleted(entry.slot)) continue;
                    auto span = aril->GetVectorSpan(entry.slot);
                    if (span.empty()) continue;
                    Metadata m;
                    (void)aril->GetMetadata(entry.slot, &m);
                    live_items[entry.id] = LiveItem{std::vector<float>(span.begin(), span.end()), std::move(m)};
                }
            }
        }
    }

    // Overlay live unpressed vectors from Rind
    if (rind_) {
        rind_->ForEachEntry([&](VectorId id, std::span<const float> vec, bool is_deleted, const Metadata* meta) {
            if (is_deleted) {
                live_items.erase(id);
            } else if (vec.size() == opt_.dim) {
                live_items[id] = LiveItem{std::vector<float>(vec.begin(), vec.end()), meta ? *meta : Metadata()};
            }
        });
    }

    std::vector<PomegranateSnapshotImpl::Item> items;
    items.reserve(live_items.size());
    for (auto& kv : live_items) {
        items.push_back({kv.first, std::move(kv.second.vec), std::move(kv.second.meta)});
    }

    *out = std::make_shared<PomegranateSnapshotImpl>(std::move(items));
    return Status::Ok();
}

Status PomegranateEngine::NewIterator(std::unique_ptr<SnapshotIterator>* out) {
    std::shared_ptr<Snapshot> snap;
    Status s = GetSnapshot(&snap);
    if (!s.ok()) return s;
    return NewIterator(snap, out);
}

Status PomegranateEngine::NewIterator(const std::shared_ptr<Snapshot>& snap,
                                     std::unique_ptr<SnapshotIterator>* out) {
    if (!snap || !out) return Status::InvalidArgument("invalid snapshot or out pointer");
    auto snap_impl = std::dynamic_pointer_cast<const PomegranateSnapshotImpl>(snap);
    if (!snap_impl) return Status::InvalidArgument("incompatible snapshot type");

    *out = std::make_unique<PomegranateSnapshotIteratorImpl>(std::move(snap_impl));
    return Status::Ok();
}

size_t PomegranateEngine::MemTableBytesUsed() const noexcept {
    return rind_ ? rind_->BytesUsed() : 0;
}

} // namespace pomai::core
