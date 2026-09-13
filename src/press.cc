// pomai/press.cc — Press: Compaction, drying, and reseeding implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "press.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>

#include "distance.h"

namespace pomai::compact {

namespace {

struct Item {
    VectorId id{0};
    std::vector<float> vec;
    Metadata meta;
};

struct PartitionedLocule {
    format::LoculeAnchor anchor;
    std::vector<Item> items;
};

std::vector<PartitionedLocule> PartitionItemsSpatially(
    std::vector<Item> all_items,
    uint32_t dim,
    MetricType metric,
    size_t locule_capacity,
    uint32_t base_id) {

    if (all_items.empty()) return {};

    if (locule_capacity == 0) locule_capacity = 50000;
    size_t num_locules = (all_items.size() + locule_capacity - 1) / locule_capacity;
    if (num_locules == 0) num_locules = 1;

    std::vector<PartitionedLocule> result(num_locules);
    for (size_t l = 0; l < num_locules; ++l) {
        result[l].anchor.id = base_id + static_cast<uint32_t>(l + 1);
        result[l].anchor.centroid.assign(dim, 0.0f);
        result[l].anchor.radius = 0.0f;
    }

    if (num_locules == 1 || all_items.size() <= num_locules) {
        // Single locule or very few items: trivial assignment
        for (size_t i = 0; i < all_items.size(); ++i) {
            size_t target_loc = std::min(i, num_locules - 1);
            result[target_loc].items.push_back(std::move(all_items[i]));
        }
        for (auto& pl : result) {
            if (pl.items.empty()) continue;
            for (const auto& it : pl.items) {
                for (size_t d = 0; d < dim; ++d) pl.anchor.centroid[d] += it.vec[d];
            }
            float inv = 1.0f / static_cast<float>(pl.items.size());
            for (size_t d = 0; d < dim; ++d) pl.anchor.centroid[d] *= inv;
            float max_dsq = 0.0f;
            for (const auto& it : pl.items) {
                float dsq = core::L2Sq(it.vec, pl.anchor.centroid);
                if (dsq > max_dsq) max_dsq = dsq;
            }
            pl.anchor.radius = std::sqrt(std::max(0.0f, max_dsq));
        }
        return result;
    }

    // Balanced K-Means clustering (K = num_locules)
    const size_t K = num_locules;
    const size_t N = all_items.size();
    std::vector<std::vector<float>> centroids(K, std::vector<float>(dim, 0.0f));

    // K-Means++ deterministic initialization (seed 42)
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> uni(0, N - 1);
    size_t first_idx = uni(rng);
    centroids[0] = all_items[first_idx].vec;

    std::vector<double> min_dist_sq(N, 1e30);
    for (size_t k = 1; k < K; ++k) {
        double total_dist = 0.0;
        for (size_t i = 0; i < N; ++i) {
            float dsq = core::L2Sq(all_items[i].vec, centroids[k - 1]);
            if (static_cast<double>(dsq) < min_dist_sq[i]) {
                min_dist_sq[i] = static_cast<double>(dsq);
            }
            total_dist += min_dist_sq[i];
        }
        if (total_dist <= 1e-9) {
            for (size_t rem = k; rem < K; ++rem) {
                centroids[rem] = all_items[rem % N].vec;
            }
            break;
        }
        std::uniform_real_distribution<double> dist_dist(0.0, total_dist);
        double r = dist_dist(rng);
        double cum = 0.0;
        size_t chosen = N - 1;
        for (size_t i = 0; i < N; ++i) {
            cum += min_dist_sq[i];
            if (cum >= r) {
                chosen = i;
                break;
            }
        }
        centroids[k] = all_items[chosen].vec;
    }

    // Normalize initial centroids if spherical
    if (metric == MetricType::kCosine || metric == MetricType::kInnerProduct) {
        for (size_t k = 0; k < K; ++k) {
            float norm_sq = 0.0f;
            for (size_t d = 0; d < dim; ++d) norm_sq += centroids[k][d] * centroids[k][d];
            if (norm_sq > 1e-12f) {
                float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (size_t d = 0; d < dim; ++d) centroids[k][d] *= inv_norm;
            }
        }
    }

    // Maximum allowed capacity per cluster (giving 25% headroom)
    size_t max_cluster_cap = (N + K - 1) / K;
    max_cluster_cap = std::min(N, static_cast<size_t>(std::ceil(max_cluster_cap * 1.25f)));

    const int kMaxIters = 12;
    std::vector<size_t> assignment(N, 0);

    struct ItemPriority {
        size_t item_idx;
        size_t best_cluster;
        float margin; // distance to second best - distance to best
    };

    for (int iter = 0; iter < kMaxIters; ++iter) {
        std::vector<ItemPriority> priorities(N);
        for (size_t i = 0; i < N; ++i) {
            size_t best_c = 0;
            float best_d = 1e30f;
            float second_best_d = 1e30f;

            for (size_t k = 0; k < K; ++k) {
                float d = 0.0f;
                if (metric == MetricType::kCosine || metric == MetricType::kInnerProduct) {
                    d = 1.0f - core::Dot(all_items[i].vec, centroids[k]);
                } else {
                    d = core::L2Sq(all_items[i].vec, centroids[k]);
                }
                if (d < best_d) {
                    second_best_d = best_d;
                    best_d = d;
                    best_c = k;
                } else if (d < second_best_d) {
                    second_best_d = d;
                }
            }
            priorities[i] = {i, best_c, second_best_d - best_d};
        }

        std::sort(priorities.begin(), priorities.end(), [](const ItemPriority& a, const ItemPriority& b) {
            return a.margin > b.margin;
        });

        std::vector<size_t> cluster_counts(K, 0);
        for (const auto& p : priorities) {
            size_t chosen_k = p.best_cluster;
            if (cluster_counts[chosen_k] >= max_cluster_cap) {
                float next_best_d = 1e30f;
                size_t next_best_k = chosen_k;
                for (size_t k = 0; k < K; ++k) {
                    if (cluster_counts[k] < max_cluster_cap) {
                        float d = (metric == MetricType::kCosine || metric == MetricType::kInnerProduct)
                            ? (1.0f - core::Dot(all_items[p.item_idx].vec, centroids[k]))
                            : core::L2Sq(all_items[p.item_idx].vec, centroids[k]);
                        if (d < next_best_d) {
                            next_best_d = d;
                            next_best_k = k;
                        }
                    }
                }
                chosen_k = next_best_k;
            }
            cluster_counts[chosen_k]++;
            assignment[p.item_idx] = chosen_k;
        }

        // Centroid update
        std::vector<std::vector<float>> new_centroids(K, std::vector<float>(dim, 0.0f));
        std::vector<size_t> new_counts(K, 0);

        for (size_t i = 0; i < N; ++i) {
            size_t c = assignment[i];
            new_counts[c]++;
            for (size_t d = 0; d < dim; ++d) {
                new_centroids[c][d] += all_items[i].vec[d];
            }
        }

        for (size_t k = 0; k < K; ++k) {
            if (new_counts[k] > 0) {
                float inv = 1.0f / static_cast<float>(new_counts[k]);
                for (size_t d = 0; d < dim; ++d) {
                    new_centroids[k][d] *= inv;
                }
                if (metric == MetricType::kCosine || metric == MetricType::kInnerProduct) {
                    float norm_sq = 0.0f;
                    for (size_t d = 0; d < dim; ++d) norm_sq += new_centroids[k][d] * new_centroids[k][d];
                    if (norm_sq > 1e-12f) {
                        float inv_norm = 1.0f / std::sqrt(norm_sq);
                        for (size_t d = 0; d < dim; ++d) new_centroids[k][d] *= inv_norm;
                    }
                }
                centroids[k] = std::move(new_centroids[k]);
            }
        }
    }

    // Distribute items into PartitionedLocules
    for (size_t i = 0; i < N; ++i) {
        size_t c = assignment[i];
        result[c].items.push_back(std::move(all_items[i]));
    }

    std::vector<PartitionedLocule> final_result;
    final_result.reserve(K);

    for (size_t k = 0; k < K; ++k) {
        if (result[k].items.empty()) continue;
        result[k].anchor.centroid = centroids[k];

        float max_dsq = 0.0f;
        for (const auto& it : result[k].items) {
            float dsq = core::L2Sq(it.vec, result[k].anchor.centroid);
            if (dsq > max_dsq) max_dsq = dsq;
        }
        result[k].anchor.radius = std::sqrt(std::max(0.0f, max_dsq));
        result[k].anchor.id = base_id + static_cast<uint32_t>(final_result.size() + 1);
        final_result.push_back(std::move(result[k]));
    }

    return final_result;
}

} // namespace

Press::Press(Env* env,
             std::string db_dir,
             uint32_t dim,
             MetricType metric,
             PressOptions options)
    : env_(env ? env : Env::Default()),
      db_dir_(std::move(db_dir)),
      dim_(dim),
      metric_(metric),
      options_(std::move(options)) {}

Press::~Press() = default;

Status Press::Compact(ingest::Rind* rind, manifest::FruitMap* fruit_map) {
    if (!fruit_map) return Status::InvalidArgument("null fruit_map");

    auto snapshot = fruit_map->CurrentSnapshot();
    std::vector<std::shared_ptr<table::MemTable>> frozen;
    if (rind) {
        frozen = rind->TakeFrozenMemtables();
    }

    if ((!snapshot || snapshot->locules().empty()) && frozen.empty()) {
        return Status::Ok();
    }

    std::map<VectorId, Item> live_items;

    // 1. Drying: extract live vectors from existing Locules in the snapshot
    if (snapshot) {
        for (const auto& loc : snapshot->locules()) {
            if (!loc) continue;
            for (const auto& aril : loc->arils()) {
                if (!aril) continue;
                for (const auto& entry : aril->directory()) {
                    if (aril->scar().IsDeleted(entry.slot)) continue;
                    auto span = aril->GetVectorSpan(entry.slot);
                    if (span.empty() || span.size() != dim_) continue;

                    Item item;
                    item.id = entry.id;
                    item.vec.assign(span.begin(), span.end());
                    (void)aril->GetMetadata(entry.slot, &item.meta);
                    live_items[item.id] = std::move(item);
                }
            }
        }
    }

    // 2. Merge frozen memtables (overwrites older records, applies deletes)
    for (const auto& table : frozen) {
        if (!table) continue;
        auto cursor = table->CreateCursor();
        table::MemTable::CursorEntry entry;
        while (cursor.Next(&entry)) {
            if (entry.is_deleted) {
                live_items.erase(entry.id);
            } else if (entry.vec.size() == dim_) {
                Item item;
                item.id = entry.id;
                item.vec.assign(entry.vec.begin(), entry.vec.end());
                if (entry.meta) item.meta = *entry.meta;
                live_items[item.id] = std::move(item);
            }
        }
    }

    uint64_t next_gen = fruit_map->NextGeneration();

    // If no live items survive
    if (live_items.empty()) {
        Status s = fruit_map->SaveManifest(next_gen, {});
        if (!s.ok()) return s;

        auto empty_snap = std::make_shared<manifest::FruitSnapshot>(
            next_gen, dim_, metric_, std::vector<std::shared_ptr<storage::Locule>>{});
        (void)fruit_map->InstallSnapshot(empty_snap);

        if (snapshot) {
            for (const auto& old_loc : snapshot->locules()) {
                if (old_loc) {
                    old_loc->CloseMapping();
                    (void)env_->DeleteFile(old_loc->filepath());
                }
            }
        }
        return Status::Ok();
    }

    // Convert map to contiguous vector
    std::vector<Item> all_items;
    all_items.reserve(live_items.size());
    for (auto& kv : live_items) {
        all_items.push_back(std::move(kv.second));
    }

    // 3. Reseeding: Spatial Locule Partitioning (Balanced Spherical/Euclidean K-Means)
    size_t locule_capacity = options_.target_aril_vector_count * options_.target_locule_aril_count;
    if (locule_capacity == 0) locule_capacity = 50000;

    auto partitioned = PartitionItemsSpatially(
        std::move(all_items), dim_, metric_, locule_capacity, 0);

    std::vector<std::string> new_locule_files;
    std::vector<std::shared_ptr<storage::Locule>> new_locules;
    new_locules.reserve(partitioned.size());
    new_locule_files.reserve(partitioned.size());

    for (auto& pl : partitioned) {
        format::LoculeAnchor anchor = std::move(pl.anchor);
        const auto& cluster_items = pl.items;
        size_t locule_count = cluster_items.size();

        // 4. Pressing: Break into Arils and build
        std::vector<std::vector<uint8_t>> serialized_arils;
        size_t aril_cap = options_.target_aril_vector_count;
        if (aril_cap == 0) aril_cap = 10000;
        size_t num_arils = (locule_count + aril_cap - 1) / aril_cap;
        if (num_arils == 0) num_arils = 1;

        for (size_t a_idx = 0; a_idx < num_arils; ++a_idx) {
            size_t a_start = a_idx * aril_cap;
            size_t a_end = std::min(locule_count, a_start + aril_cap);

            storage::ArilBuilder builder(static_cast<uint32_t>(a_idx + 1), dim_, options_.index_params, metric_);
            for (size_t i = a_start; i < a_end; ++i) {
                (void)builder.Add(cluster_items[i].id, cluster_items[i].vec, false, cluster_items[i].meta);
            }

            std::vector<uint8_t> aril_bytes;
            Status s = builder.Build(&aril_bytes);
            if (!s.ok()) return s;
            serialized_arils.push_back(std::move(aril_bytes));
        }

        std::string filename = "locule_" + std::to_string(anchor.id) + "_gen_" + std::to_string(next_gen) + ".pom";
        std::string full_path = db_dir_ + "/" + filename;

        Status s = storage::Locule::Write(env_, full_path, anchor.id, next_gen, dim_, anchor, serialized_arils);
        if (!s.ok()) return s;

        std::shared_ptr<storage::Locule> opened_loc;
        s = storage::Locule::Open(env_, full_path, &opened_loc);
        if (!s.ok()) return s;

        new_locules.push_back(std::move(opened_loc));
        new_locule_files.push_back(filename);
    }

    // 5. Atomic installation
    Status s = fruit_map->SaveManifest(next_gen, new_locule_files);
    if (!s.ok()) return s;

    auto new_snap = std::make_shared<manifest::FruitSnapshot>(
        next_gen, dim_, metric_, std::move(new_locules));
    (void)fruit_map->InstallSnapshot(new_snap);

    // 6. Purge old locules
    if (snapshot) {
        for (const auto& old_loc : snapshot->locules()) {
            if (old_loc) {
                old_loc->CloseMapping();
                (void)env_->DeleteFile(old_loc->filepath());
            }
        }
    }

    return Status::Ok();
}

Status Press::PressFrozenRindOnly(ingest::Rind* rind, manifest::FruitMap* fruit_map) {
    if (!fruit_map) return Status::InvalidArgument("null fruit_map");
    if (!rind) return Status::Ok();

    auto frozen = rind->TakeFrozenMemtables();
    if (frozen.empty()) return Status::Ok();

    std::vector<Item> items;

    for (const auto& table : frozen) {
        if (!table) continue;
        auto cursor = table->CreateCursor();
        table::MemTable::CursorEntry entry;
        while (cursor.Next(&entry)) {
            if (entry.is_deleted) continue;
            if (entry.vec.size() == dim_) {
                Item item;
                item.id = entry.id;
                item.vec.assign(entry.vec.begin(), entry.vec.end());
                if (entry.meta) item.meta = *entry.meta;
                items.push_back(std::move(item));
            }
        }
    }

    if (items.empty()) return Status::Ok();

    auto snapshot = fruit_map->CurrentSnapshot();
    uint64_t next_gen = fruit_map->NextGeneration();

    uint32_t base_id = 0;
    std::vector<std::shared_ptr<storage::Locule>> all_locules;
    std::vector<std::string> all_locule_files;
    if (snapshot) {
        base_id = static_cast<uint32_t>(snapshot->locules().size());
        for (const auto& loc : snapshot->locules()) {
            if (loc) {
                all_locules.push_back(loc);
                std::string fname = loc->filepath();
                size_t slash = fname.find_last_of("/\\");
                if (slash != std::string::npos) {
                    fname = fname.substr(slash + 1);
                }
                all_locule_files.push_back(std::move(fname));
            }
        }
    }

    size_t locule_capacity = options_.target_aril_vector_count * options_.target_locule_aril_count;
    if (locule_capacity == 0) locule_capacity = 50000;

    auto partitioned = PartitionItemsSpatially(
        std::move(items), dim_, metric_, locule_capacity, base_id);

    for (auto& pl : partitioned) {
        format::LoculeAnchor anchor = std::move(pl.anchor);
        const auto& cluster_items = pl.items;
        size_t locule_count = cluster_items.size();

        std::vector<std::vector<uint8_t>> serialized_arils;
        size_t aril_cap = options_.target_aril_vector_count;
        if (aril_cap == 0) aril_cap = 10000;
        size_t num_arils = (locule_count + aril_cap - 1) / aril_cap;
        if (num_arils == 0) num_arils = 1;

        for (size_t a_idx = 0; a_idx < num_arils; ++a_idx) {
            size_t a_start = a_idx * aril_cap;
            size_t a_end = std::min(locule_count, a_start + aril_cap);

            storage::ArilBuilder builder(static_cast<uint32_t>(a_idx + 1), dim_, options_.index_params, metric_);
            for (size_t i = a_start; i < a_end; ++i) {
                (void)builder.Add(cluster_items[i].id, cluster_items[i].vec, false, cluster_items[i].meta);
            }

            std::vector<uint8_t> aril_bytes;
            Status s = builder.Build(&aril_bytes);
            if (!s.ok()) return s;
            serialized_arils.push_back(std::move(aril_bytes));
        }

        std::string filename = "locule_" + std::to_string(anchor.id) + "_gen_" + std::to_string(next_gen) + ".pom";
        std::string full_path = db_dir_ + "/" + filename;

        Status s = storage::Locule::Write(env_, full_path, anchor.id, next_gen, dim_, anchor, serialized_arils);
        if (!s.ok()) return s;

        std::shared_ptr<storage::Locule> opened_loc;
        s = storage::Locule::Open(env_, full_path, &opened_loc);
        if (!s.ok()) return s;

        all_locules.push_back(std::move(opened_loc));
        all_locule_files.push_back(filename);
    }

    Status s = fruit_map->SaveManifest(next_gen, all_locule_files);
    if (!s.ok()) return s;

    auto new_snap = std::make_shared<manifest::FruitSnapshot>(
        next_gen, dim_, metric_, std::move(all_locules));
    (void)fruit_map->InstallSnapshot(new_snap);

    return Status::Ok();
}

} // namespace pomai::compact
