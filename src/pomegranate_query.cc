// pomai/pomegranate_query.cc — Pomegranate 5-Stage Query Orchestrator implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "pomegranate_query.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_set>

#include "distance.h"
#include "hnsw_index.h"
#include "topk.h"
#include "pomegranate_compute.h"
#include "utils/palloc_compat.h"

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

class BoundedPickQueue {
public:
    explicit BoundedPickQueue(size_t capacity = 0) : capacity_(capacity) {
        if (capacity > 0) {
            // Use palloc for inline buffer to avoid heap allocation per query
            items_data_ = static_cast<PickItem*>(palloc_malloc((capacity + 1) * sizeof(PickItem), alignof(PickItem)));
            items_capacity_ = capacity + 1;
        }
    }

    ~BoundedPickQueue() {
        if (items_data_) {
            palloc_free(items_data_);
        }
    }

    BoundedPickQueue(const BoundedPickQueue&) = delete;
    BoundedPickQueue& operator=(const BoundedPickQueue&) = delete;

    BoundedPickQueue(BoundedPickQueue&& other) noexcept
        : capacity_(other.capacity_),
          items_data_(other.items_data_),
          items_capacity_(other.items_capacity_),
          items_size_(other.items_size_) {
        other.items_data_ = nullptr;
        other.items_capacity_ = 0;
        other.items_size_ = 0;
    }

    BoundedPickQueue& operator=(BoundedPickQueue&& other) noexcept {
        if (this != &other) {
            if (items_data_) {
                palloc_free(items_data_);
            }
            capacity_ = other.capacity_;
            items_data_ = other.items_data_;
            items_capacity_ = other.items_capacity_;
            items_size_ = other.items_size_;
            other.items_data_ = nullptr;
            other.items_capacity_ = 0;
            other.items_size_ = 0;
        }
        return *this;
    }

    [[nodiscard]] size_t size() const noexcept { return items_size_; }
    [[nodiscard]] bool empty() const noexcept { return items_size_ == 0; }

    [[nodiscard]] const PickItem& top() const noexcept {
        return items_data_[0];
    }

    void pop() noexcept {
        std::pop_heap(items_data_, items_data_ + items_size_, PickMinComparator{});
        --items_size_;
    }

    void push(const PickItem& item) {
        if (items_size_ >= items_capacity_) {
            // Grow if needed (rare)
            size_t new_cap = items_capacity_ * 2;
            PickItem* new_data = static_cast<PickItem*>(palloc_malloc(new_cap * sizeof(PickItem), alignof(PickItem)));
            std::memcpy(new_data, items_data_, items_size_ * sizeof(PickItem));
            palloc_free(items_data_);
            items_data_ = new_data;
            items_capacity_ = new_cap;
        }
        items_data_[items_size_] = item;
        ++items_size_;
        std::push_heap(items_data_, items_data_ + items_size_, PickMinComparator{});
    }

    [[nodiscard]] std::vector<PickItem> ExtractItems() noexcept {
        std::vector<PickItem> result;
        result.reserve(items_size_);
        for (size_t i = 0; i < items_size_; ++i) {
            result.push_back(items_data_[i]);
        }
        items_size_ = 0;
        return result;
    }

private:
    size_t capacity_{0};
    PickItem* items_data_{nullptr};
    size_t items_capacity_{0};
    size_t items_size_{0};
};

/// Branchless 64-bit integer mixer for uniform dispersion under power-of-2 masks
inline uint64_t MixId(VectorId id) noexcept {
    uint64_t z = static_cast<uint64_t>(id) + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/// Zero-allocation open-addressing seen table.
/// Uses 512 entries directly on the stack (4KB) with linear probing and SplitMix64 hashing.
/// Completely eliminates heap node allocation and hash table rehashing on the query hot path.
class FlatSeenSet {
public:
    static constexpr size_t kInlineCap = 512;
    static constexpr uint64_t kEmpty = ~uint64_t(0);

    FlatSeenSet() noexcept {
        std::fill(inline_table_, inline_table_ + kInlineCap, kEmpty);
    }

    bool Insert(VectorId id) {
        const uint64_t h = MixId(id);
        if (heap_table_.empty()) {
            constexpr size_t mask = kInlineCap - 1;
            const size_t idx = static_cast<size_t>(h) & mask;
            for (size_t i = 0; i < kInlineCap; ++i) {
                const size_t slot = (idx + i) & mask;
                if (inline_table_[slot] == kEmpty) {
                    inline_table_[slot] = id;
                    count_++;
                    if (count_ * 4 >= kInlineCap * 3) {
                        Grow();
                    }
                    return true;
                }
                if (inline_table_[slot] == id) {
                    return false;
                }
            }
            Grow();
        }
        const size_t mask = heap_table_.size() - 1;
        const size_t idx = static_cast<size_t>(h) & mask;
        for (size_t i = 0; i < heap_table_.size(); ++i) {
            const size_t slot = (idx + i) & mask;
            if (heap_table_[slot] == kEmpty) {
                heap_table_[slot] = id;
                count_++;
                if (count_ * 4 >= heap_table_.size() * 3) {
                    Grow();
                }
                return true;
            }
            if (heap_table_[slot] == id) {
                return false;
            }
        }
        return false;
    }

private:
    void Grow() {
        size_t new_cap = heap_table_.empty() ? (kInlineCap * 4) : (heap_table_.size() * 2);
        std::vector<uint64_t> new_table(new_cap, kEmpty);
        size_t mask = new_cap - 1;

        auto rehash = [&](uint64_t val) {
            uint64_t h = MixId(static_cast<VectorId>(val));
            size_t idx = static_cast<size_t>(h) & mask;
            for (size_t i = 0; i < new_cap; ++i) {
                size_t slot = (idx + i) & mask;
                if (new_table[slot] == kEmpty) {
                    new_table[slot] = val;
                    break;
                }
            }
        };

        if (heap_table_.empty()) {
            for (size_t i = 0; i < kInlineCap; ++i) {
                if (inline_table_[i] != kEmpty) rehash(inline_table_[i]);
            }
        } else {
            for (size_t i = 0; i < heap_table_.size(); ++i) {
                if (heap_table_[i] != kEmpty) rehash(heap_table_[i]);
            }
        }
        heap_table_ = std::move(new_table);
    }

    uint64_t inline_table_[kInlineCap];
    std::vector<uint64_t> heap_table_;
    size_t count_{0};
};

} // namespace

Status PomegranateQuery::Execute(std::span<const float> query,
                                uint32_t topk,
                                const SearchOptions& opts,
                                MetricType metric,
                                const manifest::FruitSnapshot* snapshot,
                                const ingest::Rind* rind,
                                SearchHitSink& sink,
                                ptask::ThreadPool* thread_pool) {
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

    // AVX2-accelerated horizontal sum via dual-accumulator kernel (see pomegranate_compute.cc).
    // For dim=128: 8 AVX2 adds + fold + hsum vs 128 serial scalar adds.
    float query_sum = compute::SumF32(query.data(), query.size());

    size_t pick_target = (opts.ef_search > 0)
        ? std::max<size_t>(static_cast<size_t>(opts.ef_search), static_cast<size_t>(topk * 4))
        : std::max<size_t>(static_cast<size_t>(topk * 8), size_t{128});

    BoundedPickQueue pick_heap(pick_target);
    FlatSeenSet seen_ids;
    float dynamic_worst = -1e30f;

    // Helper to conditionally push into bounded pick heap with deterministic tie-breaking
    auto push_candidate = [&](const PickItem& item) {
        if (pick_heap.size() >= pick_target) {
            const auto& worst = pick_heap.top();
            if (item.score < worst.score || (item.score == worst.score && item.id >= worst.id)) {
                return;
            }
        }
        if (!seen_ids.Insert(item.id)) {
            return; // Duplicate ID
        }
        if (pick_heap.size() < pick_target) {
            pick_heap.push(item);
        } else {
            pick_heap.pop();
            pick_heap.push(item);
        }
        if (pick_heap.size() >= pick_target) {
            dynamic_worst = pick_heap.top().score;
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
        rind_hits.reserve(pick_target);
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
        const size_t total_locules = snapshot->locules().size();
        uint32_t effective_nprobe = opts.nprobe;
        if (effective_nprobe == 0) {
            effective_nprobe = opts.routing_probe_override;
        }
        if (effective_nprobe == 0 && !opts.force_fanout) {
            if (snapshot->default_nprobe() > 0) {
                effective_nprobe = snapshot->default_nprobe();
            } else {
                // Automatic fallback heuristic: max(1, round(sqrt(nlist)))
                effective_nprobe = static_cast<uint32_t>(
                    std::max<size_t>(1, static_cast<size_t>(std::round(std::sqrt(static_cast<double>(total_locules))))));
            }
        }
        auto candidate_locules = snapshot->compass().Orient(query, effective_nprobe, opts.probe_distance_ratio);

        float worst_bound = -std::numeric_limits<float>::infinity();
        if (pick_heap.size() >= pick_target) {
            worst_bound = pick_heap.top().score;
        }

        // Peel distant locules based on current worst bound
        if (worst_bound > -1e8f) {
            float peel_threshold = (metric == MetricType::kL2) ? -worst_bound : worst_bound;
            candidate_locules = snapshot->compass().Peel(candidate_locules, peel_threshold);
        }

        // ---------------------------------------------------------------------
        // Stage 3b: Intra-Locule Search (HNSW Fast-Path + 4-Way SIMD Pulp Scan)
        // ---------------------------------------------------------------------
        auto scan_locule = [&](const auto& cand_loc,
                               auto& push_fn, float& current_worst) {
            const auto& loc = cand_loc.locule;
            if (!loc) return;

            // Spatial Peel: If heap is full and lower-bound is strictly worse, skip locule
            if (current_worst > -1e29f) {
                if (metric == MetricType::kL2) {
                    if (cand_loc.min_possible_distance > -current_worst) {
                        return;
                    }
                } else {
                    if (cand_loc.min_possible_distance < current_worst) {
                        return;
                    }
                }
            }

            for (const auto& aril : loc->arils()) {
                if (!aril || aril->vector_count() == 0) continue;

                const uint32_t count = aril->vector_count();
                const auto dir = aril->directory();

                if (aril->HasGraph()) {
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
                    uint32_t aril_k = static_cast<uint32_t>(pick_target);
                    std::vector<VectorId> graph_slots;
                    std::vector<float> graph_dists;
                    graph_slots.reserve(aril_k);
                    graph_dists.reserve(aril_k);
                    int graph_default_ef = static_cast<int>(aril->local_graph()->opts().ef_search);
                    int ef_search = (opts.ef_search > 0)
                        ? std::max<int>(static_cast<int>(opts.ef_search), static_cast<int>(aril_k))
                        : std::max<int>({graph_default_ef > 0 ? graph_default_ef : 64, static_cast<int>(aril_k * 2)});
                    Status st = aril->local_graph()->Search(
                        query, aril_k, ef_search,
                        &graph_slots, &graph_dists, &filter);

                    if (st.ok() && !graph_slots.empty()) {
                        for (size_t gi = 0; gi < graph_slots.size(); ++gi) {
                            uint32_t slot = static_cast<uint32_t>(graph_slots[gi]);
                            VectorId id = (slot < dir.size()) ? dir[slot].id : 0;
                            float dist = graph_dists[gi];
                            float approx_score = (metric == MetricType::kL2) ? -dist : (1.0f - dist);

                            PickItem pi;
                            pi.id = id;
                            pi.score = approx_score;
                            pi.aril = aril.get();
                            pi.slot = slot;
                            pi.from_rind = false;
                            push_fn(pi);
                        }
                        continue;
                    }
                }

                // High-throughput 4-way fused SIMD Pulp SQ8 scan with threshold screening
                for (uint32_t slot = 0; slot < count; slot += 4) {
                    uint32_t valid = std::min<uint32_t>(4, count - slot);
                    float scores[4];
                    compute::PulpBatchScanner::Scan4(query.data(), query.size(), metric, query_sum,
                                                    aril->pulp(), slot, valid, scores);

                    const uint8_t scar_del_mask = aril->scar().IsDeleted4(slot);
                    if (scar_del_mask == 0x0F) continue; // All 4 deleted in scar

                    for (uint32_t k = 0; k < valid; ++k) {
                        float score = scores[k];
                        if (score <= current_worst) {
                            continue; // Fast threshold rejection!
                        }

                        if (scar_del_mask & (1u << k)) continue;

                        uint32_t curr_slot = slot + k;
                        VectorId id = (curr_slot < dir.size()) ? dir[curr_slot].id : 0;
                        if (rind_tombstone_snap.IsDeleted(id)) continue;

                        if (has_filters) {
                            Metadata meta;
                            (void)aril->GetMetadata(curr_slot, &meta);
                            if (!opts.Matches(meta)) continue;
                        }

                        PickItem pi;
                        pi.id = id;
                        pi.score = score;
                        pi.aril = aril.get();
                        pi.slot = curr_slot;
                        pi.from_rind = false;

                        push_fn(pi);
                    }
                }
            }
        };

        float init_worst = (pick_heap.size() >= pick_target) ? pick_heap.top().score : -1e30f;

        if (candidate_locules.size() > 1 && thread_pool && thread_pool->worker_count() > 1) {
            // Parallel locule scanning across ptask workers
            struct WorkerResult {
                std::vector<PickItem> items;
            };
            std::vector<WorkerResult> worker_results(candidate_locules.size());

            ptask::parallel_for(*thread_pool, size_t{0}, candidate_locules.size(), [&](size_t idx) {
                const auto& cand_loc = candidate_locules[idx];
                BoundedPickQueue local_heap(pick_target);
                FlatSeenSet local_seen;

                float local_worst = init_worst;
                auto local_push = [&](const PickItem& item) {
                    if (local_heap.size() >= pick_target) {
                        const auto& worst = local_heap.top();
                        if (item.score < worst.score || (item.score == worst.score && item.id >= worst.id)) {
                            return;
                        }
                    }
                    if (!local_seen.Insert(item.id)) return;
                    if (local_heap.size() < pick_target) {
                        local_heap.push(item);
                    } else {
                        local_heap.pop();
                        local_heap.push(item);
                    }
                    if (local_heap.size() >= pick_target) {
                        local_worst = local_heap.top().score;
                    }
                };

                scan_locule(cand_loc, local_push, local_worst);

                worker_results[idx].items = local_heap.ExtractItems();
            });

            // Merge worker candidates into main pick_heap
            for (auto& wr : worker_results) {
                for (const auto& item : wr.items) {
                    push_candidate(item);
                }
            }
        } else {
            // Sequential locule scan
            for (const auto& cand_loc : candidate_locules) {
                scan_locule(cand_loc, push_candidate, dynamic_worst);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Stage 4: Pick (Extract items from heap)
    // -------------------------------------------------------------------------
    std::vector<PickItem> picked = pick_heap.ExtractItems();

    // Sort candidates to enable better 4-way SIMD batching:
    // - Group by aril pointer first (same aril -> contiguous memory)
    // - from_rind false first (aril candidates), then true (rind candidates)
    // This increases the probability that 4-way batching succeeds
    std::sort(picked.begin(), picked.end(), [](const PickItem& a, const PickItem& b) {
        if (a.from_rind != b.from_rind) {
            return a.from_rind < b.from_rind; // false (aril) first, true (rind) last
        }
        if (a.aril != b.aril) {
            return a.aril < b.aril; // group by aril pointer
        }
        return a.slot < b.slot; // deterministic order within same aril
    });

    // -------------------------------------------------------------------------
    // Stage 5: Rerank (Exact distance via SeedKernel 4-Way SIMD Batching)
    // -------------------------------------------------------------------------
    std::vector<core::TopKItem> exact_hits;
    exact_hits.reserve(picked.size());

    size_t p_idx = 0;
    while (p_idx < picked.size()) {
        if (picked[p_idx].from_rind || !picked[p_idx].aril) {
            exact_hits.push_back({picked[p_idx].id, picked[p_idx].score, 0, nullptr});
            p_idx++;
            continue;
        }

        // Try to batch up to 4 contiguous candidates for 4-way SIMD reranking
        size_t batch_end = p_idx;
        const float* v_ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
        while (batch_end < picked.size() && (batch_end - p_idx) < 4) {
            if (picked[batch_end].from_rind || !picked[batch_end].aril) break;
            auto span = picked[batch_end].aril->GetVectorSpan(picked[batch_end].slot);
            if (span.empty() || span.size() != query.size()) break;
            v_ptrs[batch_end - p_idx] = span.data();
            batch_end++;
        }

        size_t batch_len = batch_end - p_idx;
        if (batch_len == 4) {
            float scores_out[4];
            compute::SeedBatchReranker::Rerank4(query.data(), query.size(), metric,
                                                v_ptrs[0], v_ptrs[1], v_ptrs[2], v_ptrs[3],
                                                scores_out);
            for (size_t k = 0; k < 4; ++k) {
                exact_hits.push_back({picked[p_idx + k].id, scores_out[k], 0, nullptr});
            }
            p_idx = batch_end;
        } else {
            for (size_t k = p_idx; k < batch_end; ++k) {
                auto span = picked[k].aril->GetVectorSpan(picked[k].slot);
                float exact_score = core::ComputeMetricScore(metric, query, span);
                exact_hits.push_back({picked[k].id, exact_score, 0, nullptr});
            }
            p_idx = batch_end;
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
                                SearchResult* out,
                                ptask::ThreadPool* thread_pool) {
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
    Status s = Execute(query, topk, opts, metric, snapshot, rind, collector, thread_pool);
    if (!s.ok()) return s;

    if (snapshot) {
        out->total_locules_count = static_cast<uint32_t>(snapshot->locules().size());
        out->routed_locules_count = out->total_locules_count;
    }
    return Status::Ok();
}

} // namespace pomai::query
