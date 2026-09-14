#include "membrane_manager.h"
#include "utils/memtable_sizing.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#include "vector_engine.h"
#include "pomai.h"
#include "iterator.h"
#include "manifest.h"
#include "utils/logging.h"

namespace pomai::core
{

using pomai::utils::CalculateDynamicMemtableThreshold;
    MembraneManager::MembraneManager(pomai::DBOptions base) : base_(std::move(base)) {}
    MembraneManager::~MembraneManager() = default;

    Status MembraneManager::Open()
    {
        auto compat = storage::Manifest::CheckCompatibility(base_.path);
        if (!compat.ok() && compat.code() != pomai::ErrorCode::kIO) {
            return compat;
        }
        opened_ = true;

        // Ensure default membrane exists and is opened.
        pomai::MembraneSpec spec;
        spec.name = std::string(kDefaultMembrane);
        spec.dim = base_.dim;
        spec.shard_count = base_.shard_count;
        spec.index_params = base_.index_params;
        spec.metric = base_.metric;

        auto st = CreateMembrane(spec);
        if (st.code() == pomai::ErrorCode::kAlreadyExists)
        {
            pomai::MembraneSpec loaded_spec;
            st = storage::Manifest::GetMembrane(base_.path, spec.name, &loaded_spec);
            if (!st.ok()) return st;

            pomai::DBOptions opt = base_;
            opt.dim = loaded_spec.dim;
            opt.shard_count = loaded_spec.shard_count;
            opt.index_params = loaded_spec.index_params;
            opt.path = base_.path + "/membranes/" + spec.name;

            auto state = std::make_shared<MembraneState>();
            state->spec = loaded_spec;
            state->vector_engine = std::make_unique<VectorEngine>(opt, loaded_spec.kind, loaded_spec.metric, loaded_spec.ttl_sec,
                                                                 loaded_spec.retention_max_count, loaded_spec.retention_max_bytes,
                                                                 loaded_spec.sync_lsn);
            state->lifecycle.SetMaxEntries(base_.max_lifecycle_entries);
            membranes_.emplace(spec.name, std::move(state));
            st = Status::Ok();
        }
        else if (!st.ok())
        {
            return st;
        }

        st = OpenMembrane(kDefaultMembrane);
        if (!st.ok()) return st;

        // Restore other membranes from manifest
        std::vector<std::string> membranes;
        st = storage::Manifest::ListMembranes(base_.path, &membranes);
        if (!st.ok()) 
        {
             return st;
        }

        for (const auto& name : membranes)
        {
            if (name == kDefaultMembrane) continue;
            
            pomai::MembraneSpec mspec;
            st = storage::Manifest::GetMembrane(base_.path, name, &mspec);
            if (!st.ok()) return st;

            if (membranes_.find(name) == membranes_.end()) {
                pomai::DBOptions opt = base_;
                opt.dim = mspec.dim;
                opt.shard_count = mspec.shard_count;
                opt.index_params = mspec.index_params;
                opt.path = base_.path + "/membranes/" + name;

                auto state = std::make_shared<MembraneState>();
                state->spec = mspec;
                state->vector_engine = std::make_unique<VectorEngine>(opt, mspec.kind, mspec.metric, mspec.ttl_sec,
                                                                     mspec.retention_max_count, mspec.retention_max_bytes,
                                                                     mspec.sync_lsn);
                state->lifecycle.SetMaxEntries(base_.max_lifecycle_entries);
                membranes_.emplace(name, std::move(state));
            }

            st = OpenMembrane(name);
            if (!st.ok()) return st;
        }

        return Status::Ok();
    }

    Status MembraneManager::Close()
    {
        return CloseAll();
    }

    Status MembraneManager::FlushAll()
    {
        // CRITICAL FIX: Snapshot membrane pointers under minimal lock, then Flush outside lock
        std::vector<std::shared_ptr<MembraneState>> membranes_snapshot;
        {
            std::shared_lock<std::shared_mutex> lock(membranes_mu_);
            membranes_snapshot.reserve(membranes_.size());
            for (auto &kv : membranes_) {
                membranes_snapshot.push_back(kv.second);
            }
        }

        // Execute Flush() entirely outside membranes_mu_ lock (blocking disk I/O)
        for (auto &state : membranes_snapshot)
        {
            if (state && state->vector_engine) {
                auto st = state->vector_engine->Flush();
                if (!st.ok())
                    return st;
            }
        }
        return Status::Ok();
    }

    Status MembraneManager::CloseAll()
    {
        std::unique_lock<std::shared_mutex> lock(membranes_mu_);
        for (auto &kv : membranes_) {
            if (kv.second && kv.second->vector_engine) {
                (void)kv.second->vector_engine->Close();
            }
        }
        membranes_.clear();
        opened_ = false;
        return Status::Ok();
    }

    std::shared_ptr<MembraneManager::MembraneState> MembraneManager::GetMembrane(std::string_view name) const
    {
        std::shared_lock<std::shared_mutex> lock(membranes_mu_);
        auto it = membranes_.find(std::string(name));
        if (it == membranes_.end())
            return nullptr;
        return it->second;
    }

    MembraneManager::MembraneState *MembraneManager::GetMembraneOrNull(std::string_view name)
    {
        std::shared_lock<std::shared_mutex> lock(membranes_mu_);
        auto it = membranes_.find(std::string(name));
        if (it == membranes_.end())
            return nullptr;
        return it->second.get();
    }

    const MembraneManager::MembraneState *MembraneManager::GetMembraneOrNull(std::string_view name) const
    {
        std::shared_lock<std::shared_mutex> lock(membranes_mu_);
        auto it = membranes_.find(std::string(name));
        if (it == membranes_.end())
            return nullptr;
        return it->second.get();
    }

    Status MembraneManager::CreateMembrane(const pomai::MembraneSpec &spec)
    {
        if (spec.name.empty())
            return Status::InvalidArgument("membrane name empty");
        if (spec.dim == 0)
            return Status::InvalidArgument("membrane dim must be > 0");
        if (spec.shard_count == 0)
            return Status::InvalidArgument("membrane shard_count must be > 0");

        std::unique_lock<std::shared_mutex> lock(membranes_mu_);
        if (membranes_.find(spec.name) != membranes_.end())
            return Status::AlreadyExists("membrane already exists");

        // 1. Persist to Manifest
        auto st = storage::Manifest::CreateMembrane(base_.path, spec);
        if (!st.ok()) return st;

        pomai::DBOptions opt = base_;
        opt.dim = spec.dim;
        opt.shard_count = spec.shard_count;
        opt.index_params = spec.index_params;
        opt.path = base_.path + "/membranes/" + spec.name;

        auto state = std::make_shared<MembraneState>();
        state->spec = spec;
        state->vector_engine = std::make_unique<VectorEngine>(opt, spec.kind, spec.metric, spec.ttl_sec, spec.retention_max_count,
                                                             spec.retention_max_bytes, spec.sync_lsn);
        state->lifecycle.SetMaxEntries(base_.max_lifecycle_entries);
        membranes_.emplace(spec.name, std::move(state));
        return Status::Ok();
    }

    Status MembraneManager::DropMembrane(std::string_view name)
    {
        std::shared_ptr<MembraneState> to_close;
        {
            std::unique_lock<std::shared_mutex> lock(membranes_mu_);
            auto it = membranes_.find(std::string(name));
            if (it == membranes_.end())
                return Status::NotFound("membrane not found");
            to_close = it->second;
            membranes_.erase(it);
        }

        // 1. Persist to Manifest
        auto st = storage::Manifest::DropMembrane(base_.path, name);
        if (!st.ok()) return st;

        // 2. Remove from Memory
        if (to_close && to_close->vector_engine) {
            (void)to_close->vector_engine->Close();
        }
        return Status::Ok();
    }

    Status MembraneManager::OpenMembrane(std::string_view name)
    {
        auto *state = GetMembraneOrNull(name);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->vector_engine) {
            return state->vector_engine->Open();
        }
        return Status::Ok();
    }

    Status MembraneManager::CloseMembrane(std::string_view name)
    {
        auto *state = GetMembraneOrNull(name);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->vector_engine) {
            return state->vector_engine->Close();
        }
        return Status::Ok();
    }

    Status MembraneManager::UpdateMembraneRetention(std::string_view name, uint32_t ttl_sec,
                                                    uint32_t retention_max_count, uint64_t retention_max_bytes) {
        auto* state = GetMembraneOrNull(name);
        if (!state) return Status::NotFound("membrane not found");
        auto st = storage::Manifest::UpdateRetentionPolicy(base_.path, name, ttl_sec, retention_max_count, retention_max_bytes);
        if (!st.ok()) return st;

        state->spec.ttl_sec = ttl_sec;
        state->spec.retention_max_count = retention_max_count;
        state->spec.retention_max_bytes = retention_max_bytes;
        return Status::Ok();
    }

    Status MembraneManager::GetMembraneRetention(std::string_view name, uint32_t* ttl_sec,
                                                 uint32_t* retention_max_count, uint64_t* retention_max_bytes) const {
        if (!ttl_sec || !retention_max_count || !retention_max_bytes) {
            return Status::InvalidArgument("retention out args must be non-null");
        }
        const auto* state = GetMembraneOrNull(name);
        if (!state) return Status::NotFound("membrane not found");
        *ttl_sec = state->spec.ttl_sec;
        *retention_max_count = state->spec.retention_max_count;
        *retention_max_bytes = state->spec.retention_max_bytes;
        return Status::Ok();
    }

    Status MembraneManager::ListMembranes(std::vector<std::string> *out) const
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        std::shared_lock<std::shared_mutex> lock(membranes_mu_);
        out->clear();
        out->reserve(membranes_.size());
        for (const auto &kv : membranes_)
            out->push_back(kv.first);
        std::sort(out->begin(), out->end());
        return Status::Ok();
    }

    Status MembraneManager::Put(std::string_view membrane, VectorId id, std::span<const float> vec)
    {
        PollMaintenance();
        return PutVector(membrane, id, vec);
    }

    Status MembraneManager::Put(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta)
    {
        PollMaintenance();
        return PutVector(membrane, id, vec, meta);
    }

    Status MembraneManager::PutVector(std::string_view membrane, VectorId id, std::span<const float> vec)
    {
        PollMaintenance();
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        auto st = MaybeApplyBackpressure(state);
        if (!st.ok()) return st;
        state->lifecycle.OnWrite(id);
        return state->vector_engine->Put(id, vec);
    }

    Status MembraneManager::PutVector(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta)
    {
        PollMaintenance();
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        auto st = MaybeApplyBackpressure(state);
        if (!st.ok()) return st;
        state->lifecycle.OnWrite(id);
        return state->vector_engine->Put(id, vec, meta);
    }

    Status MembraneManager::PutBatch(std::string_view membrane,
                                     const std::vector<VectorId>& ids,
                                     const std::vector<std::span<const float>>& vectors)
    {
        PollMaintenance();
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        
        // RAII batch guard for thread-safe burst dampening
        BatchScopeGuard batch_guard(state->active_batch_count);
        
        auto st = MaybeApplyBackpressure(state);
        if (!st.ok()) return st;
        return state->vector_engine->PutBatch(ids, vectors);
    }

    Status MembraneManager::Get(std::string_view membrane, VectorId id, std::vector<float> *out)
    {
        return Get(membrane, id, out, nullptr);
    }

    Status MembraneManager::Get(std::string_view membrane, VectorId id, std::vector<float> *out, Metadata* out_meta)
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        return state->vector_engine->Get(id, out, out_meta);
    }

    Status MembraneManager::Exists(std::string_view membrane, VectorId id, bool *exists)
    {
        if (!exists)
            return Status::InvalidArgument("exists is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        return state->vector_engine->Exists(id, exists);
    }

    Status MembraneManager::Delete(std::string_view membrane, VectorId id)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        state->lifecycle.OnDelete(id);
        return state->vector_engine->Delete(id);
    }

    Status MembraneManager::Search(std::string_view membrane, std::span<const float> query,
                                   std::uint32_t topk, pomai::SearchResult *out)
    {
        return Search(membrane, query, topk, SearchOptions{}, out);
    }

    Status MembraneManager::Search(std::string_view membrane, std::span<const float> query,
                                   std::uint32_t topk, const SearchOptions& opts, pomai::SearchResult *out)
    {
        return SearchVector(membrane, query, topk, opts, out);
    }

    Status MembraneManager::SearchVector(std::string_view membrane, std::span<const float> query,
                                         std::uint32_t topk, pomai::SearchResult *out)
    {
        return SearchVector(membrane, query, topk, SearchOptions{}, out);
    }

    Status MembraneManager::SearchVector(std::string_view membrane, std::span<const float> query,
                                         std::uint32_t topk, const SearchOptions& opts, pomai::SearchResult *out)
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        auto st = state->vector_engine->Search(query, topk, opts, out);
        if (st.ok()) for (const auto& h : out->hits) state->lifecycle.OnRead(h.id);
        return st;
    }

    Status MembraneManager::SearchVector(std::string_view membrane, std::span<const float> query,
                                         std::uint32_t topk, const SearchOptions& opts, pomai::SearchHitSink& sink)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->vector_engine) {
            return state->vector_engine->Search(query, topk, opts, sink);
        }
        return Status::NotSupported("vector_engine not available");
    }

    Status MembraneManager::SearchBatch(std::string_view membrane, std::span<const float> queries,
                                        uint32_t num_queries, std::uint32_t topk, std::vector<pomai::SearchResult> *out)
    {
        return SearchBatch(membrane, queries, num_queries, topk, SearchOptions{}, out);
    }

    Status MembraneManager::SearchBatch(std::string_view membrane, std::span<const float> queries,
                                        uint32_t num_queries, std::uint32_t topk, const SearchOptions& opts, std::vector<pomai::SearchResult> *out)
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        return state->vector_engine->SearchBatch(queries, num_queries, topk, opts, out);
    }

    Status MembraneManager::Freeze(std::string_view membrane)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        return state->vector_engine->Freeze();
    }

    Status MembraneManager::Compact(std::string_view membrane)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        return state->vector_engine->Compact();
    }

    Status MembraneManager::NewIterator(std::string_view membrane, std::unique_ptr<pomai::SnapshotIterator> *out)
    {
        if (!out) return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        return state->vector_engine->NewIterator(out);
    }

    Status MembraneManager::GetSnapshot(std::string_view membrane, std::shared_ptr<pomai::Snapshot>* out)
    {
        if (!out) return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        return state->vector_engine->GetSnapshot(out);
    }

    Status MembraneManager::NewIterator(std::string_view membrane, const std::shared_ptr<pomai::Snapshot>& snap, std::unique_ptr<pomai::SnapshotIterator> *out)
    {
        if (!out) return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->vector_engine)
            return Status::InvalidArgument("vector_engine not available");
        return state->vector_engine->NewIterator(snap, out);
    }

    Status MembraneManager::PushSync(std::string_view name, SyncReceiver* receiver)
    {
        auto *state = GetMembraneOrNull(name);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->vector_engine) return Status::InvalidArgument("vector_engine not available");

        auto st = state->vector_engine->PushSync(receiver);
        if (st.ok()) {
            uint64_t current_lsn = state->vector_engine->GetLastSyncedLSN();
            if (current_lsn > state->spec.sync_lsn) {
                state->spec.sync_lsn = current_lsn;
                (void)storage::Manifest::UpdateSyncLSN(base_.path, name, current_lsn);
            }
        }
        return st;
    }

    Status MembraneManager::MaybeApplyBackpressure(MembraneState* state)
    {
        if (!state || !state->vector_engine)
            return Status::Ok();
        if (!base_.auto_freeze_on_pressure)
            return Status::Ok();

        const std::size_t used_bytes = state->vector_engine->MemTableBytesUsed();
        
        // Calculate threshold (either manual override or dynamic)
        std::size_t threshold_bytes;
        if (base_.memtable_flush_threshold_mb > 0) {
            // Manual override mode
            threshold_bytes = static_cast<std::size_t>(base_.memtable_flush_threshold_mb) * 1024u * 1024u;
        } else {
            // Dynamic sizing mode - recalculate based on current options and dimension
            threshold_bytes = static_cast<std::size_t>(
                CalculateDynamicMemtableThreshold(base_, state->spec.dim)
            );
        }

        // Apply burst dampening if there's an active batch
        std::size_t effective_threshold = threshold_bytes;
        if (state->active_batch_count.load(std::memory_order_relaxed) > 0) {
            effective_threshold = static_cast<std::size_t>(
                threshold_bytes * (1.0f + base_.memtable_burst_dampening_factor)
            );
        }

        if (used_bytes < effective_threshold)
            return Status::Ok();

        const unsigned used_mb = static_cast<unsigned>(used_bytes / (1024u * 1024u));
        const unsigned threshold_mb = static_cast<unsigned>(threshold_bytes / (1024u * 1024u));
        POMAI_LOG_WARN("Membrane '{}' memtable pressure ({} MB / {} MB threshold). Triggering Auto-Freeze.",
                       state->spec.name, used_mb, threshold_mb);
        return state->vector_engine->Freeze();
    }

    void MembraneManager::PollMaintenance() { 
        scheduler_.PollBudget(base_.tick_max_ops, base_.tick_max_ms, base_.strict_deterministic);
    }

} // namespace pomai::core
