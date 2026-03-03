// PomaiDB Embedded: single-instance storage engine implementation.
// One Arena, one WAL, one index — strictly sequential data flow.

#include "pomai/database.h"

#include <filesystem>
#include <utility>

#include "core/distance.h"
#include "core/shard/runtime.h"
#include "core/snapshot_wrapper.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"

namespace pomai {

namespace {

constexpr std::size_t kArenaBlockBytes = 1u << 20;  // 1 MiB
constexpr std::size_t kWalSegmentBytes = 64u << 20; // 64 MiB

} // namespace

// Single-instance storage: one WAL, one MemTable, one VectorRuntime.
class StorageEngine {
public:
    StorageEngine() = default;
    ~StorageEngine() = default;

    StorageEngine(const StorageEngine&) = delete;
    StorageEngine& operator=(const StorageEngine&) = delete;

    Status Open(const EmbeddedOptions& options) {
        if (runtime_) return Status::InvalidArgument("already opened");

        std::error_code ec;
        if (!std::filesystem::exists(options.path, ec)) {
            if (!std::filesystem::create_directories(options.path, ec))
                return Status::IOError("create_directories failed");
        } else if (ec) {
            return Status::IOError("stat failed: " + ec.message());
        }

        if (options.dim == 0)
            return Status::InvalidArgument("dim must be > 0");

        auto wal = std::make_unique<storage::Wal>(
            options.path, 0u, kWalSegmentBytes, options.fsync);
        auto st = wal->Open();
        if (!st.ok()) return st;

        auto mem = std::make_unique<table::MemTable>(
            options.dim, kArenaBlockBytes);
        st = wal->ReplayInto(*mem);
        if (!st.ok()) return st;

        std::string data_dir = options.path;
        auto rt = std::make_unique<core::VectorRuntime>(
            0u, std::move(data_dir), options.dim,
            MembraneKind::kVector, options.metric,
            std::move(wal), std::move(mem), options.index_params);
        runtime_ = std::move(rt);
        return runtime_->Start();
    }

    void Close() { runtime_.reset(); }

    Status Append(VectorId id, std::span<const float> vec) {
        return runtime_ ? runtime_->Put(id, vec)
                       : Status::InvalidArgument("not opened");
    }

    Status Append(VectorId id, std::span<const float> vec,
                  const Metadata& meta) {
        return runtime_ ? runtime_->Put(id, vec, meta)
                        : Status::InvalidArgument("not opened");
    }

    Status AppendBatch(const std::vector<VectorId>& ids,
                       const std::vector<std::span<const float>>& vectors) {
        return runtime_ ? runtime_->PutBatch(ids, vectors)
                        : Status::InvalidArgument("not opened");
    }

    Status Get(VectorId id, std::vector<float>* out,
               Metadata* out_meta) {
        return runtime_ ? runtime_->Get(id, out, out_meta)
                        : Status::InvalidArgument("not opened");
    }

    Status Exists(VectorId id, bool* exists) {
        return runtime_ ? runtime_->Exists(id, exists)
                       : Status::InvalidArgument("not opened");
    }

    Status Delete(VectorId id) {
        return runtime_ ? runtime_->Delete(id)
                        : Status::InvalidArgument("not opened");
    }

    Status Flush() {
        return runtime_ ? runtime_->Flush()
                        : Status::InvalidArgument("not opened");
    }

    Status Freeze() {
        return runtime_ ? runtime_->Freeze()
                        : Status::InvalidArgument("not opened");
    }

    Status GetSnapshot(std::shared_ptr<Snapshot>* out) {
        if (!runtime_) return Status::InvalidArgument("not opened");
        if (!out) return Status::InvalidArgument("output is null");
        auto internal = runtime_->GetSnapshot();
        *out = std::make_shared<core::SnapshotWrapper>(std::move(internal));
        return Status::Ok();
    }

    Status NewIterator(const std::shared_ptr<Snapshot>& snap,
                       std::unique_ptr<SnapshotIterator>* out) {
        if (!runtime_) return Status::InvalidArgument("not opened");
        if (!out) return Status::InvalidArgument("output is null");
        if (!snap) return Status::InvalidArgument("snapshot is null");
        auto* wrapper = dynamic_cast<core::SnapshotWrapper*>(snap.get());
        if (!wrapper) return Status::InvalidArgument("invalid snapshot type");
        return runtime_->NewIterator(wrapper->GetInternal(), out);
    }

    Status Search(std::span<const float> query, std::uint32_t topk,
                  const SearchOptions& opts,
                  SearchResult* out) {
        if (!runtime_) return Status::InvalidArgument("not opened");
        if (!out) return Status::InvalidArgument("output is null");
        out->Clear();
        out->hits.clear();
        std::vector<SearchHit> hits;
        auto st = runtime_->Search(query, topk, opts, &hits);
        if (!st.ok()) return st;
        out->hits = std::move(hits);
        out->routed_shards_count = 1;
        out->routed_buckets_count = 0;
        return Status::Ok();
    }

    Status SearchBatch(std::span<const float> queries, std::uint32_t num_queries,
                      std::uint32_t topk, const SearchOptions& opts,
                      std::vector<SearchResult>* out) {
        if (!runtime_) return Status::InvalidArgument("not opened");
        if (!out) return Status::InvalidArgument("output is null");
        out->clear();
        out->resize(num_queries);
        if (num_queries == 0 || topk == 0) return Status::Ok();
        std::vector<uint32_t> indices(num_queries);
        for (uint32_t i = 0; i < num_queries; ++i) indices[i] = i;
        std::vector<std::vector<SearchHit>> per_query;
        auto st = runtime_->SearchBatchLocal(queries, indices, topk, opts, &per_query);
        if (!st.ok()) return st;
        for (uint32_t i = 0; i < num_queries; ++i) {
            (*out)[i].hits = std::move(per_query[i]);
            (*out)[i].routed_shards_count = 1;
        }
        return Status::Ok();
    }

private:
    std::unique_ptr<core::VectorRuntime> runtime_;
};

// -----------------------------------------------------------------------------
// Database
// -----------------------------------------------------------------------------

Database::Database() = default;

Database::~Database() {
    (void)Close();
}

Status Database::Open(const EmbeddedOptions& options) {
    if (opened_) return Status::InvalidArgument("already opened");
    if (options.path.empty())
        return Status::InvalidArgument("path empty");
    if (options.dim == 0)
        return Status::InvalidArgument("dim must be > 0");

    core::InitDistance();
    storage_engine_ = std::make_unique<StorageEngine>();
    auto st = storage_engine_->Open(options);
    if (!st.ok()) {
        storage_engine_.reset();
        return st;
    }
    opened_ = true;
    return Status::Ok();
}

Status Database::Close() {
    if (!opened_) return Status::Ok();
    if (storage_engine_) storage_engine_->Close();
    storage_engine_.reset();
    opened_ = false;
    return Status::Ok();
}

Status Database::Flush() {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Flush();
}

Status Database::Freeze() {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Freeze();
}

Status Database::GetSnapshot(std::shared_ptr<Snapshot>* out) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->GetSnapshot(out);
}

Status Database::NewIterator(const std::shared_ptr<Snapshot>& snap,
                             std::unique_ptr<SnapshotIterator>* out) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->NewIterator(snap, out);
}

Status Database::AddVector(VectorId id, std::span<const float> vec) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Append(id, vec);
}

Status Database::AddVector(VectorId id, std::span<const float> vec,
                          const Metadata& meta) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Append(id, vec, meta);
}

Status Database::AddVectorBatch(
    const std::vector<VectorId>& ids,
    const std::vector<std::span<const float>>& vectors) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->AppendBatch(ids, vectors);
}

Status Database::Get(VectorId id, std::vector<float>* out) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Get(id, out, nullptr);
}

Status Database::Get(VectorId id, std::vector<float>* out,
                    Metadata* out_meta) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Get(id, out, out_meta);
}

Status Database::Exists(VectorId id, bool* exists) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Exists(id, exists);
}

Status Database::Delete(VectorId id) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Delete(id);
}

Status Database::Search(std::span<const float> query,
                        std::uint32_t topk,
                        SearchResult* out) {
    return Search(query, topk, SearchOptions{}, out);
}

Status Database::Search(std::span<const float> query,
                        std::uint32_t topk,
                        const SearchOptions& opts,
                        SearchResult* out) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Search(query, topk, opts, out);
}

Status Database::SearchBatch(std::span<const float> queries,
                             std::uint32_t num_queries,
                             std::uint32_t topk,
                             const SearchOptions& opts,
                             std::vector<SearchResult>* out) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->SearchBatch(queries, num_queries, topk, opts, out);
}

} // namespace pomai
