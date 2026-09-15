#include "vector_engine.h"

#include <algorithm>

#include "pomegranate_engine.h"
#include "distance.h"
#include "sync_provider.h"
#include "storage/palloc_io.h"
#include "vulkan_device_context.h"

namespace pomai::core {

VectorEngine::VectorEngine(pomai::DBOptions opt,
                           pomai::MembraneKind kind,
                           pomai::MetricType metric,
                           uint32_t ttl_sec,
                           uint32_t retention_max_count,
                           uint64_t retention_max_bytes,
                           uint64_t sync_lsn)
    : opt_(std::move(opt)),
      kind_(kind),
      metric_(metric),
      ttl_sec_(ttl_sec),
      retention_max_count_(retention_max_count),
      retention_max_bytes_(retention_max_bytes),
      sync_lsn_(sync_lsn) {}

VectorEngine::~VectorEngine() = default;

Status VectorEngine::EnsureOpen() const {
    if (!opened_ || !engine_) {
        return Status::Corruption("VectorEngine is not open");
    }
    return Status::Ok();
}

Status VectorEngine::ValidateVector(std::span<const float> vec) const {
    if (vec.size() != opt_.dim) {
        return Status::InvalidArgument("Vector dimension mismatch");
    }
    return Status::Ok();
}

Status VectorEngine::Open() {
    core::InitDistance();
    return OpenLocked();
}

Status VectorEngine::OpenLocked() {
    if (kind_ != pomai::MembraneKind::kVector) {
        return Status::InvalidArgument("VectorEngine only supports VECTOR membranes");
    }
    if (opt_.dim == 0) {
        return Status::InvalidArgument("VectorEngine requires dim > 0");
    }

    // Create directory using palloc filesystem
    Status st_env = storage::PallocFilesystem::CreateDir(opt_.path.c_str());
    if (!st_env.ok()) {
        return Status::IOError("VectorEngine CreateDir failed: " + opt_.path + " (" + st_env.message() + ")");
    }

    engine_ = std::make_unique<PomegranateEngine>(opt_, metric_, nullptr);
    Status s = engine_->Open();
    if (!s.ok()) {
        engine_.reset();
        return s;
    }

    opened_ = true;
    return Status::Ok();
}

Status VectorEngine::Close() {
    if (!opened_) return Status::Ok();
    if (engine_) {
        (void)engine_->Close();
        engine_.reset();
    }
    opened_ = false;
    return Status::Ok();
}

Status VectorEngine::Put(VectorId id, std::span<const float> vec) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    s = ValidateVector(vec);
    if (!s.ok()) return s;
    return engine_->Put(id, vec);
}

Status VectorEngine::Put(VectorId id,
                         std::span<const float> vec,
                         const pomai::Metadata& meta) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    s = ValidateVector(vec);
    if (!s.ok()) return s;
    return engine_->Put(id, vec, meta);
}

Status VectorEngine::PutBatch(const std::vector<VectorId>& ids,
                              const std::vector<std::span<const float>>& vectors) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->PutBatch(ids, vectors);
}

Status VectorEngine::PutBatch(const std::vector<VectorId>& ids,
                              const std::vector<std::vector<float>>& vectors) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->PutBatch(ids, vectors);
}

Status VectorEngine::PutBatch(std::span<const VectorId> ids,
                              std::span<const float> vectors,
                              std::size_t dimension) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->PutBatch(ids, vectors, dimension);
}

Status VectorEngine::Get(VectorId id, std::vector<float>* out) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->Get(id, out);
}

Status VectorEngine::Get(VectorId id,
                         std::vector<float>* out,
                         pomai::Metadata* out_meta) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->Get(id, out, out_meta);
}

Status VectorEngine::Exists(VectorId id, bool* exists) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->Exists(id, exists);
}

Status VectorEngine::Delete(VectorId id) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->Delete(id);
}

Status VectorEngine::Flush() {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->Flush();
}

Status VectorEngine::Freeze() {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->Freeze();
}

Status VectorEngine::Compact() {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->Compact();
}

Status VectorEngine::PushSync(SyncReceiver* receiver) {
    if (!receiver) return Status::InvalidArgument("receiver is null");
    if (engine_) {
        (void)engine_->Flush();
    }
    WalStreamer streamer(opt_.path, 0);
    uint64_t next_lsn = sync_lsn_;
    auto st = streamer.PushSince(sync_lsn_, receiver, &next_lsn);
    if (st.ok()) {
        sync_lsn_ = next_lsn;
    }
    return st;
}

uint64_t VectorEngine::GetLastSyncedLSN() const {
    return sync_lsn_;
}

std::size_t VectorEngine::MemTableBytesUsed() const noexcept {
    return engine_ ? engine_->MemTableBytesUsed() : 0;
}

Status VectorEngine::GetSnapshot(std::shared_ptr<pomai::Snapshot>* out) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->GetSnapshot(out);
}

Status VectorEngine::NewIterator(std::unique_ptr<pomai::SnapshotIterator>* out) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->NewIterator(out);
}

Status VectorEngine::NewIterator(const std::shared_ptr<pomai::Snapshot>& snap,
                                 std::unique_ptr<pomai::SnapshotIterator>* out) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->NewIterator(snap, out);
}

Status VectorEngine::Search(std::span<const float> query,
                            std::uint32_t topk,
                            pomai::SearchResult* out) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->Search(query, topk, out);
}

Status VectorEngine::Search(std::span<const float> query,
                            std::uint32_t topk,
                            std::vector<pomai::SearchHit>* out) {
    if (!out) return Status::InvalidArgument("null out pointer");
    SearchResult sr;
    Status s = Search(query, topk, &sr);
    if (!s.ok()) return s;
    *out = std::move(sr.hits);
    return Status::Ok();
}

Status VectorEngine::Search(std::span<const float> query,
                            std::uint32_t topk,
                            const SearchOptions& opts,
                            pomai::SearchResult* out) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->Search(query, topk, opts, out);
}

Status VectorEngine::Search(std::span<const float> query,
                            std::uint32_t topk,
                            const SearchOptions& opts,
                            std::vector<pomai::SearchHit>* out) {
    if (!out) return Status::InvalidArgument("null out pointer");
    SearchResult sr;
    Status s = Search(query, topk, opts, &sr);
    if (!s.ok()) return s;
    *out = std::move(sr.hits);
    return Status::Ok();
}

Status VectorEngine::Search(std::span<const float> query,
                            std::uint32_t topk,
                            const SearchOptions& opts,
                            pomai::SearchHitSink& sink) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->Search(query, topk, opts, sink);
}

Status VectorEngine::SearchBatch(std::span<const float> queries,
                                 std::uint32_t num_queries,
                                 std::uint32_t topk,
                                 std::vector<pomai::SearchResult>* out) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->SearchBatch(queries, num_queries, topk, {}, out);
}

Status VectorEngine::SearchBatch(std::span<const float> queries,
                                 std::uint32_t num_queries,
                                 std::uint32_t topk,
                                 const SearchOptions& opts,
                                 std::vector<pomai::SearchResult>* out) {
    Status s = EnsureOpen();
    if (!s.ok()) return s;
    return engine_->SearchBatch(queries, num_queries, topk, opts, out);
}

} // namespace pomai::core
