#include "database.h"
#include "options.h"
#include "status.h"
#include "hooks.h"
#include "env.h"
#include "scheduler.h"
#include "vector_engine.h"
#include "memtable.h"
#include "wal.h"
#include "internal_engine.h"
#include "vector_pod.h"
#include "metadata.h"
#include "search.h"
#include "palloc_compat.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>

namespace pomai {
namespace {
constexpr uint32_t kKernelHotPathMaxMsgs = 8;
constexpr uint32_t kKernelHotPathMaxMs = 2;
}

Status StorageEngine::Open(const EmbeddedOptions& options) {
    auto env = options.env ? options.env : Env::Default();
    auto v_path = options.path + "/vectors";

    DBOptions dopt;
    dopt.path = v_path;
    dopt.env = env;
    dopt.dim = options.dim;
    dopt.metric = options.metric;
    dopt.fsync = options.fsync;
    dopt.index_params = options.index_params;
    dopt.memtable_flush_threshold_mb = options.memtable_flush_threshold_mb;

    auto v_engine = std::make_unique<core::VectorEngine>(
        dopt, MembraneKind::kVector, options.metric);
    Status st = v_engine->Open();
    if (!st.ok()) return st;

    return kernel_.RegisterPod(std::make_unique<core::VectorPod>(std::move(v_engine)));
}

void StorageEngine::Close() {
    kernel_.Stop();
}

Status StorageEngine::Flush() {
    Status st = Status::Ok();
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kFlush);
    msg.result_ptr = &st;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);
    return st;
}

Status StorageEngine::Freeze() {
    Status st = Status::Ok();
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kFreeze);
    msg.result_ptr = &st;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);
    return st;
}

Status StorageEngine::Append(VectorId id, std::span<const float> vec, const Metadata& meta) {
    Status st = Status::Ok();
    struct P {
        VectorId id;
        const float* vec_data;
        size_t vec_size;
        const Metadata* meta;
    } p = {id, vec.data(), vec.size(), &meta};

    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kPutWithMeta, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&p), sizeof(P)));
    msg.result_ptr = &st;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);

    if (st.ok()) {
        for (auto& h : hooks_) h->OnPostPut(id, vec, meta);
    }
    return st;
}

Status StorageEngine::Append(VectorId id, std::span<const float> vec) {
    return Append(id, vec, Metadata());
}

Status StorageEngine::AppendBatch(const std::vector<VectorId>& ids, const std::vector<std::span<const float>>& vectors) {
    Status st = Status::Ok();
    struct P {
        const std::vector<VectorId>* ids;
        const std::vector<std::span<const float>>* vectors;
    } payload = {&ids, &vectors};

    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kPutBatch, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&payload), sizeof(payload)));
    msg.result_ptr = &st;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);

    if (st.ok()) {
        for (size_t i = 0; i < ids.size(); ++i) {
            for (auto& h : hooks_) h->OnPostPut(ids[i], vectors[i], Metadata());
        }
    }
    return st;
}

Status StorageEngine::Get(VectorId id, std::vector<float>* out, Metadata* meta) {
    Status st = Status::Ok();
    struct P {
        VectorId id;
        std::vector<float>* out_vec;
        Metadata* out_meta;
    } p = {id, out, meta};

    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kGetWithMeta, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&p), sizeof(P)));
    msg.result_ptr = &st;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);
    return st;
}

Status StorageEngine::Exists(VectorId id, bool* exists) {
    Status st = Status::Ok();
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kExists, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&id), sizeof(id)));
    msg.result_ptr = exists;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);
    return st;
}

Status StorageEngine::Delete(VectorId id) {
    Status st = Status::Ok();
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kDelete, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&id), sizeof(id)));
    msg.result_ptr = &st;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);
    return st;
}

Status StorageEngine::Search(std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out) {
    Status st = Status::Ok();
    struct P {
        uint32_t topk;
        const float* query_data;
        size_t query_size;
        const SearchOptions* opts;
    } p = {topk, query.data(), query.size(), &opts};
    struct SearchResultEnvelope {
        SearchResult* out;
        Status* st;
    } env{out, &st};

    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kSearch, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&p), sizeof(P)));
    msg.result_ptr = &env;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);
    return st;
}

Status StorageEngine::GetSnapshot(std::shared_ptr<Snapshot>* out) {
    Status st = Status::Ok();
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kGetSnapshot);
    msg.result_ptr = out;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);
    return st;
}

Status StorageEngine::NewIterator(const std::shared_ptr<Snapshot>& snap, std::unique_ptr<SnapshotIterator>* out) {
    Status st = Status::Ok();
    const std::shared_ptr<Snapshot>* s_ptr = &snap;
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kNewIterator, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&s_ptr), sizeof(void*)));
    msg.result_ptr = out;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);
    return st;
}

Status StorageEngine::PushSync(core::SyncReceiver* receiver) {
    Status st = Status::Ok();
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kSync);
    msg.result_ptr = receiver;
    msg.status_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    (void)kernel_.ProcessBudget(kKernelHotPathMaxMsgs, kKernelHotPathMaxMs);
    return st;
}

std::size_t StorageEngine::GetMemTableBytesUsed() const {
    auto* pod = static_cast<core::VectorPod*>(const_cast<core::MicroKernel&>(kernel_).GetPod(core::PodId::kIndex));
    return pod ? pod->GetMemTableBytesUsed() : 0;
}

void StorageEngine::AddPostPutHook(std::shared_ptr<PostPutHook> hook) {
    hooks_.push_back(std::move(hook));
}

// Tasks
class SyncTask : public core::DatabaseTask {
public:
    SyncTask(StorageEngine* engine, std::shared_ptr<core::SyncReceiver> receiver)
        : engine_(engine), receiver_(std::move(receiver)) {}
    Status Run() override { return engine_->PushSync(receiver_.get()); }
    std::string Name() const override { return "SyncTask"; }
private:
    StorageEngine* engine_;
    std::shared_ptr<core::SyncReceiver> receiver_;
};

class MaintenanceTask : public core::DatabaseTask {
public:
    explicit MaintenanceTask(Database* db) : db_(db) {}
    Status Run() override { return db_->MaybeApplyBackpressure(); }
    std::string Name() const override { return "Maintenance"; }
private:
    Database* db_;
};

struct Database::Impl {
    core::TaskScheduler scheduler;
    std::shared_ptr<core::SyncReceiver> sync_receiver;
};

Database::Database() : opened_(false), impl_(std::make_unique<Impl>()) {}
Database::~Database() { (void)Close(); }

Status Database::Open(const EmbeddedOptions& options) {
    pomai::util::EnsurePallocInitialized();
    if (opened_) return Status::InvalidArgument("already open");
    if (options.dim == 0) return Status::InvalidArgument("dimension must be greater than 0");
    if (options.path.empty()) return Status::InvalidArgument("path cannot be empty");
    
    std::uint32_t max_mb = options.max_memtable_mb;
    if (max_mb == 0) max_mb = 128u; // Default
    max_memtable_bytes_ = max_mb * 1024ULL * 1024ULL;
    
    std::uint8_t threshold_pct = options.pressure_threshold_percent;
    if (threshold_pct == 0) threshold_pct = 80u;
    pressure_threshold_bytes_ = (max_memtable_bytes_ * threshold_pct) / 100u;
    
    auto_freeze_on_pressure_ = options.auto_freeze_on_pressure;

    storage_engine_ = std::make_unique<StorageEngine>();
    auto st = storage_engine_->Open(options);
    if (!st.ok()) return st;

    opened_ = true;
    if (impl_->sync_receiver) {
        impl_->scheduler.RegisterPeriodic(std::make_unique<SyncTask>(storage_engine_.get(), impl_->sync_receiver), std::chrono::seconds(10));
    }
    impl_->scheduler.RegisterPeriodic(std::make_unique<MaintenanceTask>(this), std::chrono::seconds(5));
    
    return Status::Ok();
}

Status Database::Close() {
    if (!opened_) return Status::Ok();
    storage_engine_->Close();
    storage_engine_.reset();
    opened_ = false;
    return Status::Ok();
}

Status Database::Flush() { return opened_ ? storage_engine_->Flush() : Status::InvalidArgument("closed"); }
Status Database::Freeze() { return opened_ ? storage_engine_->Freeze() : Status::InvalidArgument("closed"); }

Status Database::TryFreezeIfPressured() {
    if (!opened_) return Status::InvalidArgument("closed");
    if (GetMemTableBytesUsed() > pressure_threshold_bytes_) {
        return Freeze();
    }
    return Status::Ok();
}

Status Database::MaybeApplyBackpressure() {
    if (!opened_) return Status::InvalidArgument("closed");
    if (GetMemTableBytesUsed() > pressure_threshold_bytes_) {
        if (auto_freeze_on_pressure_) {
            return Freeze();
        } else {
            return Status::ResourceExhausted("memtable pressure");
        }
    }
    return Status::Ok();
}

std::size_t Database::GetMemTableBytesUsed() const {
    return opened_ ? storage_engine_->GetMemTableBytesUsed() : 0;
}

Status Database::AddVector(VectorId id, std::span<const float> vec) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    st = storage_engine_->Append(id, vec);
    impl_->scheduler.Poll();
    return st;
}

Status Database::AddVector(VectorId id, std::span<const float> vec, const Metadata& meta) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    st = storage_engine_->Append(id, vec, meta);
    impl_->scheduler.Poll();
    return st;
}

Status Database::AddVectorBatch(const std::vector<VectorId>& ids, const std::vector<std::span<const float>>& vectors) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    st = storage_engine_->AppendBatch(ids, vectors);
    impl_->scheduler.Poll();
    return st;
}

Status Database::PutBatch(const std::vector<VectorId>& ids, const std::vector<std::vector<float>>& vectors) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    if (ids.size() != vectors.size()) return Status::InvalidArgument("mismatch");
    std::vector<std::span<const float>> spans;
    for (const auto& v : vectors) spans.push_back(v);
    st = storage_engine_->AppendBatch(ids, spans);
    impl_->scheduler.Poll();
    return st;
}

Status Database::PutBatch(std::span<const VectorId> ids, std::span<const float> vectors, std::size_t dimension) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    if (dimension == 0 || ids.empty() || vectors.size() != ids.size() * dimension) return Status::InvalidArgument("invalid args");
    std::vector<VectorId> owned_ids(ids.begin(), ids.end());
    std::vector<std::span<const float>> spans;
    for (size_t i = 0; i < ids.size(); ++i) spans.emplace_back(vectors.data() + i * dimension, dimension);
    st = storage_engine_->AppendBatch(owned_ids, spans);
    impl_->scheduler.Poll();
    return st;
}

Status Database::Get(VectorId id, std::vector<float>* out) { 
    if (!out) return Status::InvalidArgument("out cannot be null");
    return opened_ ? storage_engine_->Get(id, out, nullptr) : Status::InvalidArgument("closed"); 
}
Status Database::Get(VectorId id, std::vector<float>* out, Metadata* meta) { 
    if (!out) return Status::InvalidArgument("out cannot be null");
    return opened_ ? storage_engine_->Get(id, out, meta) : Status::InvalidArgument("closed"); 
}

Status Database::Exists(VectorId id, bool* exists) {
    return opened_ ? storage_engine_->Exists(id, exists) : Status::InvalidArgument("closed");
}

Status Database::Delete(VectorId id) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = storage_engine_->Delete(id);
    impl_->scheduler.Poll();
    return st;
}

Status Database::Search(std::span<const float> query, uint32_t topk, SearchResult* out) {
    if (!out) return Status::InvalidArgument("out cannot be null");
    return Search(query, topk, SearchOptions(), out);
}

Status Database::Search(std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out) {
    if (!out) return Status::InvalidArgument("out cannot be null");
    return opened_ ? storage_engine_->Search(query, topk, opts, out) : Status::InvalidArgument("closed");
}

Status Database::SearchBatch(std::span<const float> queries, uint32_t num_queries, uint32_t topk, const SearchOptions& opts, std::vector<SearchResult>* out) {
    if (!opened_) return Status::InvalidArgument("closed");
    if (!out) return Status::InvalidArgument("out null");
    out->resize(num_queries);
    size_t dim = queries.size() / num_queries;
    for (uint32_t i = 0; i < num_queries; ++i) {
        auto st = storage_engine_->Search(queries.subspan(i * dim, dim), topk, opts, &(*out)[i]);
        if (!st.ok()) return st;
    }
    return Status::Ok();
}

Status Database::GetSnapshot(std::shared_ptr<Snapshot>* out) {
    if (!out) return Status::InvalidArgument("out cannot be null");
    return opened_ ? storage_engine_->GetSnapshot(out) : Status::InvalidArgument("closed");
}

Status Database::NewIterator(const std::shared_ptr<Snapshot>& snap, std::unique_ptr<SnapshotIterator>* out) {
    return opened_ ? storage_engine_->NewIterator(snap, out) : Status::InvalidArgument("closed");
}

void Database::RegisterSyncReceiver(std::shared_ptr<core::SyncReceiver> receiver) {
    impl_->sync_receiver = std::move(receiver);
    if (opened_ && storage_engine_) {
        impl_->scheduler.RegisterPeriodic(std::make_unique<SyncTask>(storage_engine_.get(), impl_->sync_receiver), std::chrono::seconds(10));
    }
}

void Database::AddPostPutHook(std::shared_ptr<PostPutHook> hook) {
    if (opened_) storage_engine_->AddPostPutHook(std::move(hook));
}

} // namespace pomai

