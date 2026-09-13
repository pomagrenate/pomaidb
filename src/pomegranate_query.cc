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
#include "hnsw_index.h"
#include "topk.h"

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
        if (a.score != b.score) {
            return a.score > b.score; // min-heap: smallest score on top
        }
        return a.id < b.id; // tie-break: larger id on top (evicted first)
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
    if (topk == 0) {
        return Status::Ok();
    }
    if (query.empty()) {
        return Status::InvalidArgument("query vector is empty");
    }
    for (float v : query) {
        if (!std::isfinite(v)) {
            return Status::InvalidArgument("query vector contains non-finite values (NaN or Inf)");
        }
    }
    if (rind && rind->dimension() > 0 && query.size() != rind->dimension()) {
        return Status::InvalidArgument("query vector dimension mismatch: expected " +
                                       std::to_string(rind->dimension()) + ", got " +
                                       std::to_string(query.size()));
    }
    if (snapshot && snapshot->dimension() > 0 && query.size() != snapshot->dimension()) {
        return Status::InvalidArgument("query vector dimension mismatch: expected " +
                                       std::to_string(snapshot->dimension()) + ", got " +
                                       std::to_string(query.size()));
    }

    float query_sum = 0.0f;
    for (float v : query) {
        query_sum += v;
    }

    size_t pick_target = std::max<size_t>(topk * 2, 64);

    std::priority_queue<PickItem, std::vector<PickItem>, PickMinComparator> pick_heap;
    std::unordered_set<VectorId> seen_ids;
    seen_ids.reserve(pick_target * 2);

    // Helper to conditionally push into bounded pick heap with deterministic tie-breaking
    auto push_candidate = [&](const PickItem& item) {
        if (pick_heap.size() >= pick_target) {
            const auto& worst = pick_heap.top();
            if (item.score < worst.score || (item.score == worst.score && item.id >= worst.id)) {
                return;
            }
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

    // Capture point-in-time tombstone snapshot to eliminate lock contention during scan loops
    const auto rind_tombstone_snap = rind ? rind->CaptureTombstoneSnapshot() : ingest::RindTombstoneSnapshot{};

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
            PickItem pi;
            pi.id = rh.id;
            pi.score = rh.distance;
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
        // Stage 3b: Intra-Locule Search (HNSW Fast-Path + Pulp SQ8 Fallback)
        // ---------------------------------------------------------------------
        for (const auto& cand_loc : candidate_locules) {
            const auto& loc = cand_loc.locule;
            if (!loc) continue;

            // Dynamic Spatial Peel: If pick_heap is full and this locule's lower-bound distance
            // is strictly worse than the worst score in the heap, skip this locule
            if (pick_heap.size() >= pick_target) {
                float worst_score = pick_heap.top().score;
                if (metric == MetricType::kL2) {
                    if (cand_loc.min_possible_distance > -worst_score) {
                        continue;
                    }
                } else {
                    if (cand_loc.min_possible_distance < worst_score) {
                        continue;
                    }
                }
            }

            for (const auto& aril : loc->arils()) {
                if (!aril || aril->vector_count() == 0) continue;

                const uint32_t count = aril->vector_count();
                const auto dir = aril->directory();

                if (aril->HasGraph()) {
                    // Intra-Locule HNSW Graph Search with in-graph filtering
                    class ArilFilter : public index::IdFilter {
                    public:
                        ArilFilter(const storage::ArilReader* a,
                                   const ingest::RindTombstoneSnapshot& r_snap,
                                   bool h_filt,
                                   const SearchOptions& o)
                            : aril_(a), rind_snap_(r_snap), has_filters_(h_filt), opts_(o) {}

                        bool IsAllowed(VectorId slot_as_id) override {
                            uint32_t slot = static_cast<uint32_t>(slot_as_id);
                            if (aril_->scar().IsDeleted(slot)) return false;
                            const auto d = aril_->directory();
                            VectorId id = (slot < d.size()) ? d[slot].id : 0;
                            if (rind_snap_.IsDeleted(id)) return false;
                            if (has_filters_) {
                                Metadata meta;
                                (void)aril_->GetMetadata(slot, &meta);
                                if (!opts_.Matches(meta)) return false;
                            }
                            return true;
                        }
                    private:
                        const storage::ArilReader* aril_;
                        const ingest::RindTombstoneSnapshot& rind_snap_;
                        bool has_filters_;
                        const SearchOptions& opts_;
                    };

                    ArilFilter filter(aril.get(), rind_tombstone_snap, has_filters, opts);
                    std::vector<VectorId> graph_slots;
                    std::vector<float> graph_dists;
                    uint32_t aril_k = static_cast<uint32_t>(std::min<size_t>(pick_target, topk + 48));
                    int ef_search = std::max<int>({static_cast<int>(opts.ef_search), static_cast<int>(aril_k * 6), 512});
                    Status st = aril->local_graph()->Search(
                        query, aril_k, ef_search,
                        &graph_slots, &graph_dists, &filter);

                    if (st.ok() && !graph_slots.empty()) {
                        for (size_t i = 0; i < graph_slots.size(); ++i) {
                            uint32_t slot = static_cast<uint32_t>(graph_slots[i]);
                            VectorId id = (slot < dir.size()) ? dir[slot].id : 0;
                            float dist = graph_dists[i];
                            float approx_score = (metric == MetricType::kL2) ? -dist : (1.0f - dist);

                            PickItem pi;
                            pi.id = id;
                            pi.score = approx_score;
                            pi.aril = aril.get();
                            pi.slot = slot;
                            pi.from_rind = false;
                            push_candidate(pi);
                        }
                        continue; // Successfully retrieved candidates via HNSW
                    }
                }

                // Fallback path: SIMD Pulp SQ8 flat scan
                for (uint32_t slot = 0; slot < count; ++slot) {
                    if (aril->scar().IsDeleted(slot)) continue;

                    VectorId id = (slot < dir.size()) ? dir[slot].id : 0;
                    if (rind_tombstone_snap.IsDeleted(id)) continue;

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
    std::vector<core::TopKItem> exact_hits;
    exact_hits.reserve(picked.size());

    for (const auto& item : picked) {
        if (item.from_rind) {
            exact_hits.push_back({item.id, item.score, 0, nullptr});
        } else if (item.aril) {
            auto span = item.aril->GetVectorSpan(item.slot);
            if (!span.empty() && span.size() == query.size()) {
                float exact_score = core::ComputeMetricScore(metric, query, span);
                exact_hits.push_back({item.id, exact_score, 0, nullptr});
            }
        }
    }

    // Optimal deterministic top-k selection
    core::SelectTopK(exact_hits, topk);

    for (const auto& hit : exact_hits) {
        sink.Push(hit.id, hit.score);
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
