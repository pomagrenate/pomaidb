// pomai/pomegranate_query.cc — Pomegranate 5-Stage Query Orchestrator implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "pomegranate_query.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <unordered_set>

#include "distance.h"

namespace pomai::query {

namespace {

struct PickItem {
    VectorId id{0};
    float score{0.0f};
    const storage::ArilReader* aril{nullptr};
    uint32_t slot{0};
    bool from_rind{false};
};

struct PickMinComparator {
    bool operator()(const PickItem& a, const PickItem& b) const noexcept {
        return a.score > b.score; // min-heap: smallest score on top
    }
};

} // namespace

Status PomegranateQuery::Execute(std::span<const float> query,
                                uint32_t topk,
                                const SearchOptions& opts,
                                MetricType metric,
                                const manifest::FruitSnapshot* snapshot,
                                const ingest::Rind* rind,
                                SearchHitSink& sink) {
    if (topk == 0 || query.empty()) {
        return Status::Ok();
    }

    float query_sum = 0.0f;
    for (float v : query) {
        query_sum += v;
    }

    size_t pick_target = std::max<size_t>(topk * 4, 64);

    std::priority_queue<PickItem, std::vector<PickItem>, PickMinComparator> pick_heap;
    std::unordered_set<VectorId> seen_ids;
    seen_ids.reserve(pick_target * 2);

    // Helper to conditionally push into bounded pick heap
    auto push_candidate = [&](const PickItem& item) {
        if (pick_heap.size() >= pick_target && item.score <= pick_heap.top().score) {
            return;
        }
        if (!seen_ids.insert(item.id).second) {
            return; // Duplicate ID
        }
        if (pick_heap.size() < pick_target) {
            pick_heap.push(item);
        } else {
            pick_heap.pop();
            pick_heap.push(item);
        }
    };

    const bool has_filters = !opts.filters.empty() || opts.as_of_ts > 0 || opts.as_of_lsn > 0 ||
                             !opts.partition_device_id.empty() || !opts.partition_location_id.empty();

    // -------------------------------------------------------------------------
    // Stage 3a: Taste Rind (immediate RAM visibility for live memtables)
    // -------------------------------------------------------------------------
    if (rind) {
        std::vector<ingest::RindHit> rind_hits;
        rind->Taste(query, static_cast<uint32_t>(pick_target), metric, &rind_hits);
        for (const auto& rh : rind_hits) {
            if (has_filters) {
                Metadata meta;
                (void)rind->Get(rh.id, nullptr, &meta);
                if (!opts.Matches(meta)) continue;
            }
            float score = (metric == MetricType::kL2) ? -rh.distance : rh.distance;
            PickItem pi;
            pi.id = rh.id;
            pi.score = score;
            pi.from_rind = true;
            push_candidate(pi);
        }
    }

    // -------------------------------------------------------------------------
    // Stage 1 & 2: Orient & Peel
    // -------------------------------------------------------------------------
    if (snapshot && !snapshot->locules().empty()) {
        auto candidate_locules = snapshot->compass().Orient(query, opts.routing_probe_override);

        float worst_bound = -std::numeric_limits<float>::infinity();
        if (pick_heap.size() >= pick_target) {
            worst_bound = pick_heap.top().score;
        }

        // Peel distant locules based on current worst bound
        if (metric == MetricType::kL2 && worst_bound > -1e8f) {
            candidate_locules = snapshot->compass().Peel(candidate_locules, -worst_bound);
        }

        // ---------------------------------------------------------------------
        // Stage 3b: Taste Pulp in Arils
        // ---------------------------------------------------------------------
        for (const auto& cand_loc : candidate_locules) {
            const auto& loc = cand_loc.locule;
            if (!loc) continue;

            for (const auto& aril : loc->arils()) {
                if (!aril || aril->vector_count() == 0) continue;

                const uint32_t count = aril->vector_count();
                const auto dir = aril->directory();

                for (uint32_t slot = 0; slot < count; ++slot) {
                    if (aril->scar().IsDeleted(slot)) continue;

                    VectorId id = (slot < dir.size()) ? dir[slot].id : 0;
                    if (rind && rind->IsDeleted(id)) continue;

                    if (has_filters) {
                        Metadata meta;
                        (void)aril->GetMetadata(slot, &meta);
                        if (!opts.Matches(meta)) continue;
                    }

                    float score = aril->pulp().Taste(query, slot, metric, query_sum);

                    PickItem pi;
                    pi.id = id;
                    pi.score = score;
                    pi.aril = aril.get();
                    pi.slot = slot;
                    pi.from_rind = false;

                    push_candidate(pi);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Stage 4: Pick (Extract items from heap)
    // -------------------------------------------------------------------------
    std::vector<PickItem> picked;
    picked.reserve(pick_heap.size());
    while (!pick_heap.empty()) {
        picked.push_back(pick_heap.top());
        pick_heap.pop();
    }

    // -------------------------------------------------------------------------
    // Stage 5: Rerank (Exact distance via SeedKernel)
    // -------------------------------------------------------------------------
    struct ExactHit {
        VectorId id{0};
        float score{0.0f};
    };
    std::vector<ExactHit> exact_hits;
    exact_hits.reserve(picked.size());

    for (const auto& item : picked) {
        if (item.from_rind) {
            exact_hits.push_back({item.id, item.score});
        } else if (item.aril) {
            auto span = item.aril->GetVectorSpan(item.slot);
            if (!span.empty() && span.size() == query.size()) {
                float exact_score = 0.0f;
                if (metric == MetricType::kL2) {
                    exact_score = -core::L2Sq(query, span);
                } else {
                    exact_score = core::Dot(query, span);
                }
                exact_hits.push_back({item.id, exact_score});
            }
        }
    }

    std::sort(exact_hits.begin(), exact_hits.end(), [](const ExactHit& a, const ExactHit& b) {
        return a.score > b.score; // highest score first
    });

    size_t out_count = std::min<size_t>(topk, exact_hits.size());
    for (size_t i = 0; i < out_count; ++i) {
        sink.Push(exact_hits[i].id, exact_hits[i].score);
    }

    return Status::Ok();
}

Status PomegranateQuery::Execute(std::span<const float> query,
                                uint32_t topk,
                                const SearchOptions& opts,
                                MetricType metric,
                                const manifest::FruitSnapshot* snapshot,
                                const ingest::Rind* rind,
                                SearchResult* out) {
    if (!out) return Status::InvalidArgument("null search result output");
    out->Clear();

    class VectorHitCollector final : public SearchHitSink {
    public:
        explicit VectorHitCollector(std::vector<SearchHit>* hits) : hits_(hits) {}
        void Push(VectorId id, float score) override {
            hits_->push_back({id, score});
        }
    private:
        std::vector<SearchHit>* hits_;
    };

    VectorHitCollector collector(&out->hits);
    Status s = Execute(query, topk, opts, metric, snapshot, rind, collector);
    if (!s.ok()) return s;

    if (snapshot) {
        out->total_shards_count = static_cast<uint32_t>(snapshot->locules().size());
        out->routed_shards_count = out->total_shards_count;
    }
    return Status::Ok();
}

} // namespace pomai::query
