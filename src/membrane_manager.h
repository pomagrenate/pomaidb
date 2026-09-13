#pragma once
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "options.h"
#include "search.h"
#include "status.h"
#include "iterator.h"
#include "metadata.h"
#include "snapshot.h"
#include "semantic_lifecycle.h"
#include "scheduler.h"

namespace pomai::core
{

    class VectorEngine;
    class SyncReceiver;

    class MembraneManager
    {
    public:
        explicit MembraneManager(pomai::DBOptions base);
        ~MembraneManager();

        MembraneManager(const MembraneManager &) = delete;
        MembraneManager &operator=(const MembraneManager &) = delete;

        Status Open();
        Status Close();

        Status FlushAll();
        Status CloseAll();

        Status CreateMembrane(const pomai::MembraneSpec &spec);
        Status DropMembrane(std::string_view name);
        Status OpenMembrane(std::string_view name);
        Status CloseMembrane(std::string_view name);
        Status UpdateMembraneRetention(std::string_view name, uint32_t ttl_sec, uint32_t retention_max_count, uint64_t retention_max_bytes);
        Status GetMembraneRetention(std::string_view name, uint32_t* ttl_sec, uint32_t* retention_max_count, uint64_t* retention_max_bytes) const;

        Status ListMembranes(std::vector<std::string> *out) const;

        Status Put(std::string_view membrane, VectorId id, std::span<const float> vec);
        Status Put(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta);
        Status PutVector(std::string_view membrane, VectorId id, std::span<const float> vec);
        Status PutVector(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta);
        Status PutBatch(std::string_view membrane,
                        const std::vector<VectorId>& ids,
                        const std::vector<std::span<const float>>& vectors);
        Status Get(std::string_view membrane, VectorId id, std::vector<float> *out);
        Status Get(std::string_view membrane, VectorId id, std::vector<float> *out, Metadata* out_meta);
        Status Exists(std::string_view membrane, VectorId id, bool *exists);
        Status Delete(std::string_view membrane, VectorId id);

        Status Search(std::string_view membrane, std::span<const float> query, std::uint32_t topk, pomai::SearchResult *out);
        Status Search(std::string_view membrane, std::span<const float> query, std::uint32_t topk, const SearchOptions& opts, pomai::SearchResult *out);
        Status SearchVector(std::string_view membrane, std::span<const float> query, std::uint32_t topk, pomai::SearchResult *out);
        Status SearchVector(std::string_view membrane, std::span<const float> query, std::uint32_t topk, const SearchOptions& opts, pomai::SearchResult *out);
        /** @brief Zero-copy search vector directly into a sink. */
        Status SearchVector(std::string_view membrane, std::span<const float> query, std::uint32_t topk, const SearchOptions& opts, pomai::SearchHitSink& sink);
        Status SearchBatch(std::string_view membrane, std::span<const float> queries, uint32_t num_queries, std::uint32_t topk, std::vector<pomai::SearchResult>* out);
        Status SearchBatch(std::string_view membrane, std::span<const float> queries, uint32_t num_queries, std::uint32_t topk, const SearchOptions& opts, std::vector<pomai::SearchResult>* out);

        Status Freeze(std::string_view membrane);
        Status Compact(std::string_view membrane);
        Status NewIterator(std::string_view membrane, std::unique_ptr<pomai::SnapshotIterator> *out);
        Status GetSnapshot(std::string_view name, std::shared_ptr<pomai::Snapshot> *out);
        Status NewIterator(std::string_view membrane, const std::shared_ptr<pomai::Snapshot>& snap, std::unique_ptr<pomai::SnapshotIterator> *out);
        Status PushSync(std::string_view name, SyncReceiver* receiver);

        const pomai::DBOptions& GetOptions() const { return base_; }

        // Default membrane convenience: use name "__default__"
        static constexpr std::string_view kDefaultMembrane = "__default__";

    private:
        struct MembraneState
        {
            pomai::MembraneSpec spec;
            std::unique_ptr<VectorEngine> vector_engine;
            SemanticLifecycle lifecycle;
        };

        MembraneState *GetMembraneOrNull(std::string_view name);
        const MembraneState *GetMembraneOrNull(std::string_view name) const;
        std::shared_ptr<MembraneState> GetMembrane(std::string_view name) const;

        /** Backpressure helper: if enabled and over threshold, Freeze() before writes. */
        Status MaybeApplyBackpressure(MembraneState* state);
        void PollMaintenance();

        pomai::DBOptions base_;
        bool opened_ = false;

        mutable std::shared_mutex membranes_mu_;
        std::unordered_map<std::string, std::shared_ptr<MembraneState>> membranes_;
        TaskScheduler scheduler_;
    };

} // namespace pomai::core
