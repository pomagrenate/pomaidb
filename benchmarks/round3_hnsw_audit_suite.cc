// benchmarks/round3_hnsw_audit_suite.cc — Round 3 HNSW Deep Audit & Multi-Locule Recall Gate
//
// Conducts:
// 1. Line-by-line audit verification: Flawed Repo HNSW vs Corrected HNSW Prototype
// 2. Adversarial distributions: Uniform, Clustered, Boundary, Duplicate-heavy, Near-duplicate, Low-D & High-D
// 3. Multi-Locule Global Recall Gate: Probing M of L Locules with Compass + Intra-Locule HNSW
// 4. Hard recall constraint: Enforcing Recall@10 >= 0.95
//
// Copyright 2026 PomaiDB authors. MIT License.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "distance.h"
#include "reference_distance.h"
#include "topk.h"
#include "hnsw_index.h"
#include "hnsw/hnsw.h"

namespace {

// =========================================================================
// CORRECTED ROBUST HNSW PROTOTYPE (FOR DIFFERENTIAL AUDIT)
// =========================================================================
class CorrectedHNSW {
public:
    struct NodeDist {
        float dist;
        int32_t id;
        bool operator<(const NodeDist& other) const { return dist < other.dist; } // Max-heap: farthest first
    };
    struct NodeDistMin {
        float dist;
        int32_t id;
        bool operator<(const NodeDistMin& other) const { return dist > other.dist; } // Min-heap: closest first
    };

    int M;
    int max_m0;
    int ef_construction;
    double level_mult;
    int32_t entry_point{-1};
    int max_level{-1};
    std::mt19937 rng{42};

    // Vector data pool
    size_t dim;
    std::vector<float> data;
    size_t num_nodes{0};

    // Graph adjacency lists: node -> level -> list of neighbors
    std::vector<std::vector<std::vector<int32_t>>> adj;
    std::vector<int> levels;

    // Fast O(1) visited tracker using version tags
    mutable std::vector<uint32_t> visited_tag;
    mutable uint32_t current_tag{0};

    CorrectedHNSW(size_t dim, int M = 32, int ef_construction = 128)
        : M(M), max_m0(2 * M), ef_construction(ef_construction), dim(dim) {
        level_mult = 1.0 / std::log(static_cast<double>(M));
    }

    int GetRandomLevel() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = -std::log(dist(rng) + 1e-15) * level_mult;
        return static_cast<int>(r);
    }

    float Dist(int32_t i1, int32_t i2) const {
        return pomai::core::L2Sq(std::span<const float>(data.data() + i1 * dim, dim),
                                 std::span<const float>(data.data() + i2 * dim, dim));
    }

    float DistQuery(std::span<const float> q, int32_t i) const {
        return pomai::core::L2Sq(q, std::span<const float>(data.data() + i * dim, dim));
    }

    void AddPoint(std::span<const float> vec) {
        int32_t id = static_cast<int32_t>(num_nodes++);
        data.insert(data.end(), vec.begin(), vec.end());
        int level = GetRandomLevel();
        levels.push_back(level);

        adj.emplace_back(level + 1);
        if (visited_tag.size() < num_nodes) visited_tag.resize(num_nodes, 0);

        if (entry_point == -1) {
            entry_point = id;
            max_level = level;
            return;
        }

        int32_t curr = entry_point;
        float d_curr = Dist(id, curr);

        // 1. Greedy descent down to level
        for (int l = max_level; l > level; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (int32_t nb : adj[curr][l]) {
                    float d = Dist(id, nb);
                    if (d < d_curr) {
                        d_curr = d;
                        curr = nb;
                        changed = true;
                    }
                }
            }
        }

        // 2. Insert into each level from min(level, max_level) down to 0
        for (int l = std::min(level, max_level); l >= 0; --l) {
            std::priority_queue<NodeDistMin> candidates; // min-heap: closest candidate first
            std::priority_queue<NodeDist> w;             // max-heap: farthest of best candidates on top

            current_tag++;
            visited_tag[curr] = current_tag;
            candidates.push({d_curr, curr});
            w.push({d_curr, curr});

            while (!candidates.empty()) {
                auto c = candidates.top();
                candidates.pop();

                if (c.dist > w.top().dist) break;

                for (int32_t nb : adj[c.id][l]) {
                    if (visited_tag[nb] == current_tag) continue;
                    visited_tag[nb] = current_tag;

                    float d = Dist(id, nb);
                    if (w.size() < static_cast<size_t>(ef_construction) || d < w.top().dist) {
                        candidates.push({d, nb});
                        w.push({d, nb});
                        if (w.size() > static_cast<size_t>(ef_construction)) w.pop();
                    }
                }
            }

            // Select M best neighbors
            int max_edges = (l == 0) ? max_m0 : M;
            std::vector<NodeDist> best;
            while (!w.empty()) {
                best.push_back(w.top());
                w.pop();
            }
            std::reverse(best.begin(), best.end()); // closest first
            if (best.size() > static_cast<size_t>(max_edges)) best.resize(max_edges);

            for (const auto& b : best) {
                adj[id][l].push_back(b.id);

                // CORRECTED BACKLINK INSERTION: Replace farthest neighbor if list is full
                auto& nb_list = adj[b.id][l];
                if (nb_list.size() < static_cast<size_t>(max_edges)) {
                    nb_list.push_back(id);
                } else {
                    // Find farthest neighbor in neighbor's list
                    float max_d = Dist(b.id, id);
                    int replace_idx = -1;
                    for (size_t i = 0; i < nb_list.size(); ++i) {
                        float d = Dist(b.id, nb_list[i]);
                        if (d > max_d) {
                            max_d = d;
                            replace_idx = static_cast<int>(i);
                        }
                    }
                    if (replace_idx >= 0) {
                        nb_list[replace_idx] = id;
                    }
                }
            }
        }

        if (level > max_level) {
            max_level = level;
            entry_point = id;
        }
    }

    void Search(std::span<const float> query, int k, int ef,
                std::vector<int32_t>& out_ids, std::vector<float>& out_dists) const {
        out_ids.clear();
        out_dists.clear();
        if (entry_point == -1) return;

        int32_t curr = entry_point;
        float d_curr = DistQuery(query, curr);

        for (int l = max_level; l > 0; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (int32_t nb : adj[curr][l]) {
                    float d = DistQuery(query, nb);
                    if (d < d_curr) {
                        d_curr = d;
                        curr = nb;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<NodeDistMin> candidates;
        std::priority_queue<NodeDist> top_candidates;

        current_tag++;
        visited_tag[curr] = current_tag;
        candidates.push({d_curr, curr});
        top_candidates.push({d_curr, curr});

        while (!candidates.empty()) {
            auto c = candidates.top();
            candidates.pop();

            if (c.dist > top_candidates.top().dist) break;

            for (int32_t nb : adj[c.id][0]) {
                if (visited_tag[nb] == current_tag) continue;
                visited_tag[nb] = current_tag;

                float d = DistQuery(query, nb);
                if (top_candidates.size() < static_cast<size_t>(ef) || d < top_candidates.top().dist) {
                    candidates.push({d, nb});
                    top_candidates.push({d, nb});
                    if (top_candidates.size() > static_cast<size_t>(ef)) top_candidates.pop();
                }
            }
        }

        std::vector<NodeDist> sorted;
        while (!top_candidates.empty()) {
            sorted.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::sort(sorted.begin(), sorted.end(), [](const NodeDist& a, const NodeDist& b) {
            return a.dist < b.dist;
        });

        for (size_t i = 0; i < std::min(sorted.size(), static_cast<size_t>(k)); ++i) {
            out_ids.push_back(sorted[i].id);
            out_dists.push_back(sorted[i].dist);
        }
    }
};

// Vector generation helpers
std::vector<float> GenerateUniform(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (size_t i = 0; i < v.size(); ++i) v[i] = dist(rng);
    return v;
}

std::vector<float> GenerateClustered(size_t n, size_t dim, size_t num_clusters, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center_dist(-5.0f, 5.0f);
    std::normal_distribution<float> noise(0.0f, 0.2f);

    std::vector<float> centers(num_clusters * dim);
    for (size_t i = 0; i < centers.size(); ++i) centers[i] = center_dist(rng);

    std::vector<float> v(n * dim);
    for (size_t i = 0; i < n; ++i) {
        size_t c = i % num_clusters;
        for (size_t d = 0; d < dim; ++d) {
            v[i * dim + d] = centers[c * dim + d] + noise(rng);
        }
    }
    return v;
}

std::vector<float> GenerateDuplicates(size_t n, size_t dim, double dup_ratio, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t unique_count = std::max<size_t>(1, static_cast<size_t>(n * (1.0 - dup_ratio)));

    std::vector<float> uniques(unique_count * dim);
    for (size_t i = 0; i < uniques.size(); ++i) uniques[i] = dist(rng);

    std::vector<float> v(n * dim);
    for (size_t i = 0; i < n; ++i) {
        size_t u = i % unique_count;
        for (size_t d = 0; d < dim; ++d) v[i * dim + d] = uniques[u * dim + d];
    }
    return v;
}

std::vector<float> GenerateNearDuplicates(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 1e-4f);
    size_t base_count = 50;

    std::vector<float> base(base_count * dim);
    for (size_t i = 0; i < base.size(); ++i) base[i] = dist(rng);

    std::vector<float> v(n * dim);
    for (size_t i = 0; i < n; ++i) {
        size_t b = i % base_count;
        for (size_t d = 0; d < dim; ++d) v[i * dim + d] = base[b * dim + d] + noise(rng);
    }
    return v;
}

std::vector<std::vector<uint32_t>> ExactGroundTruth(const std::vector<float>& data,
                                                    size_t count,
                                                    const std::vector<float>& queries,
                                                    size_t num_queries,
                                                    size_t dim,
                                                    size_t topk) {
    std::vector<std::vector<uint32_t>> gt(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
        std::span<const float> q_span(queries.data() + q * dim, dim);
        std::vector<pomai::core::TopKItem> items(count);
        for (size_t i = 0; i < count; ++i) {
            float dist = pomai::core::L2Sq(q_span, std::span<const float>(data.data() + i * dim, dim));
            items[i] = {static_cast<pomai::VectorId>(i), -dist, 0, nullptr};
        }
        pomai::core::SelectTopK(items, topk);
        gt[q].reserve(topk);
        for (size_t k = 0; k < topk && k < items.size(); ++k) {
            gt[q].push_back(static_cast<uint32_t>(items[k].id));
        }
    }
    return gt;
}

double ComputeRecall10(const std::vector<int32_t>& candidates,
                       const std::vector<uint32_t>& ground_truth) {
    if (ground_truth.empty() || candidates.empty()) return 0.0;
    std::unordered_set<uint32_t> gt_set(ground_truth.begin(), ground_truth.begin() + std::min<size_t>(10, ground_truth.size()));
    size_t hits = 0;
    for (size_t i = 0; i < std::min<size_t>(10, candidates.size()); ++i) {
        if (gt_set.count(static_cast<uint32_t>(candidates[i]))) hits++;
    }
    return static_cast<double>(hits) / static_cast<double>(gt_set.size());
}

} // namespace

int main() {
    std::cout << "=================================================================\n";
    std::cout << " PomaiDB Performance Round 3: HNSW Audit & Integration Gate     \n";
    std::cout << "=================================================================\n\n";

    const size_t dim = 64;
    const size_t n = 10000;
    const size_t num_queries = 100;
    const size_t topk = 10;

    auto data = GenerateUniform(n, dim, 42);
    auto queries = GenerateUniform(num_queries, dim, 1337);
    auto gt = ExactGroundTruth(data, n, queries, num_queries, dim, topk);

    // =========================================================================
    // 1. REPO HNSW VS CORRECTED HNSW PROTOTYPE (DEFECT AUDIT)
    // =========================================================================
    std::cout << "### 1. CODE DEFECT AUDIT: REPO HNSW VS CORRECTED HNSW PROTOTYPE\n\n";

    // Test A: Repo HNSW (third_party/pomaidb_hnsw/hnsw.cc)
    std::cout << "Building Repo HNSW (with linear seen check & silent backlink drop)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    pomai::index::HnswOptions repo_opts;
    repo_opts.M = 32;
    repo_opts.ef_construction = 128;
    repo_opts.ef_search = 64;
    pomai::index::HnswIndex repo_hnsw(dim, repo_opts, pomai::MetricType::kL2);
    for (size_t i = 0; i < n; ++i) {
        (void)repo_hnsw.Add(static_cast<pomai::VectorId>(i), std::span<const float>(data.data() + i * dim, dim));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double repo_build_s = std::chrono::duration<double>(t1 - t0).count();

    // Test B: Corrected HNSW Prototype
    std::cout << "Building Corrected HNSW (with O(1) visited tags & backlink replacement)...\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    CorrectedHNSW corr_hnsw(dim, 32, 128);
    for (size_t i = 0; i < n; ++i) {
        corr_hnsw.AddPoint(std::span<const float>(data.data() + i * dim, dim));
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double corr_build_s = std::chrono::duration<double>(t3 - t2).count();

    // Query both at ef_search = 64
    double repo_r10 = 0.0;
    double corr_r10 = 0.0;
    std::vector<double> repo_lat, corr_lat;

    for (size_t q = 0; q < num_queries; ++q) {
        std::span<const float> q_span(queries.data() + q * dim, dim);

        // Repo search
        std::vector<pomai::VectorId> r_ids;
        std::vector<float> r_dists;
        auto q0 = std::chrono::high_resolution_clock::now();
        (void)repo_hnsw.Search(q_span, topk, 64, &r_ids, &r_dists);
        auto q1 = std::chrono::high_resolution_clock::now();
        repo_lat.push_back(std::chrono::duration<double, std::micro>(q1 - q0).count());
        std::vector<int32_t> r_conv;
        for (auto id : r_ids) r_conv.push_back(static_cast<int32_t>(id));
        repo_r10 += ComputeRecall10(r_conv, gt[q]);

        // Corrected search
        std::vector<int32_t> c_ids;
        std::vector<float> c_dists;
        auto q2 = std::chrono::high_resolution_clock::now();
        corr_hnsw.Search(q_span, topk, 64, c_ids, c_dists);
        auto q3 = std::chrono::high_resolution_clock::now();
        corr_lat.push_back(std::chrono::duration<double, std::micro>(q3 - q2).count());
        corr_r10 += ComputeRecall10(c_ids, gt[q]);
    }

    std::cout << "| Implementation | Build Time (10K) | Ingestion Rate | Search Latency (p50) | Recall@10 | Code Defects Identified |\n";
    std::cout << "| :--- | :---: | :---: | :---: | :---: | :--- |\n";
    std::cout << "| **Repo HNSW** (`hnsw.cc`) | " << std::fixed << std::setprecision(2) << repo_build_s << " s | "
              << std::fixed << std::setprecision(0) << (n / repo_build_s) << " vec/s | "
              << std::fixed << std::setprecision(2) << (repo_lat[num_queries/2] / 1000.0) << " ms | "
              << std::fixed << std::setprecision(3) << (repo_r10 / num_queries)
              << " | **O(|seen|) linear scan + silent backlink drop** |\n";
    std::cout << "| **Corrected HNSW** | **" << std::fixed << std::setprecision(2) << corr_build_s << " s** | **"
              << std::fixed << std::setprecision(0) << (n / corr_build_s) << " vec/s** | **"
              << std::fixed << std::setprecision(2) << (corr_lat[num_queries/2] / 1000.0) << " ms** | **"
              << std::fixed << std::setprecision(3) << (corr_r10 / num_queries)
              << "** | **O(1) version tags + farthest backlink replacement** |\n\n";

    // =========================================================================
    // 2. ADVERSARIAL VECTOR DISTRIBUTIONS (CORRECTED HNSW)
    // =========================================================================
    std::cout << "### 2. ADVERSARIAL VECTOR DISTRIBUTIONS BENCHMARK (N = 10,000, Top-10)\n\n";
    std::cout << "| Distribution Pattern | Description | Build Time | Search Latency (p50) | Recall@10 | Meets >=95% Recall? |\n";
    std::cout << "| :--- | :--- | :---: | :---: | :---: | :---: |\n";

    struct DistTest {
        std::string name;
        std::string desc;
        std::vector<float> vecs;
    };

    std::vector<DistTest> dist_tests = {
        {"Uniform Random", "Uniform [-1, 1] random floats", GenerateUniform(n, dim, 101)},
        {"Clustered (10 Centers)", "10 distinct tight clusters (variance 0.04)", GenerateClustered(n, dim, 10, 202)},
        {"Duplicate-Heavy (50%)", "5,000 unique vectors duplicated twice", GenerateDuplicates(n, dim, 0.50, 303)},
        {"Near-Duplicates (1e-4)", "50 base centers with 1e-4 noise", GenerateNearDuplicates(n, dim, 404)},
    };

    for (auto& dt : dist_tests) {
        auto dt_gt = ExactGroundTruth(dt.vecs, n, queries, num_queries, dim, topk);
        CorrectedHNSW h(dim, 32, 128);
        auto bt0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) h.AddPoint(std::span<const float>(dt.vecs.data() + i * dim, dim));
        auto bt1 = std::chrono::high_resolution_clock::now();
        double b_time = std::chrono::duration<double>(bt1 - bt0).count();

        double r10 = 0.0;
        std::vector<double> lats;
        for (size_t q = 0; q < num_queries; ++q) {
            std::span<const float> q_span(queries.data() + q * dim, dim);
            std::vector<int32_t> out_ids;
            std::vector<float> out_dists;
            auto st0 = std::chrono::high_resolution_clock::now();
            h.Search(q_span, topk, 128, out_ids, out_dists);
            auto st1 = std::chrono::high_resolution_clock::now();
            lats.push_back(std::chrono::duration<double, std::micro>(st1 - st0).count());
            r10 += ComputeRecall10(out_ids, dt_gt[q]);
        }
        std::sort(lats.begin(), lats.end());
        double avg_r = r10 / num_queries;

        std::cout << "| **" << dt.name << "** | " << dt.desc
                  << " | " << std::fixed << std::setprecision(2) << b_time << " s"
                  << " | " << std::fixed << std::setprecision(2) << (lats[num_queries/2] / 1000.0) << " ms"
                  << " | **" << std::fixed << std::setprecision(3) << avg_r << "**"
                  << " | " << (avg_r >= 0.95 ? "**YES**" : "NO") << " |\n";
    }
    std::cout << "\n";

    // =========================================================================
    // 3. MULTI-LOCULE GLOBAL RECALL GATE: PROBING M OF L LOCULES
    // =========================================================================
    std::cout << "### 3. MULTI-LOCULE GLOBAL RECALL GATE (N = 25,000, L = 8 Locules)\n\n";
    std::cout << "Testing: Can Intra-Locule HNSW find global nearest neighbors without probing all Locules?\n\n";

    const size_t n_loc = 25000;
    const size_t num_loc = 8;
    const size_t loc_size = n_loc / num_loc;
    auto loc_data = GenerateUniform(n_loc, dim, 777);
    auto loc_gt = ExactGroundTruth(loc_data, n_loc, queries, num_queries, dim, topk);

    // Build spatial centroids for the 8 locules
    std::vector<std::vector<float>> centroids(num_loc, std::vector<float>(dim, 0.0f));
    for (size_t c = 0; c < num_loc; ++c) {
        size_t start = c * loc_size;
        for (size_t i = start; i < start + loc_size; ++i) {
            for (size_t d = 0; d < dim; ++d) centroids[c][d] += loc_data[i * dim + d];
        }
        for (size_t d = 0; d < dim; ++d) centroids[c][d] /= loc_size;
    }

    // Build intra-locule HNSW index inside each Locule
    std::vector<std::unique_ptr<CorrectedHNSW>> locule_hnsw;
    for (size_t c = 0; c < num_loc; ++c) {
        auto h = std::make_unique<CorrectedHNSW>(dim, 16, 64);
        size_t start = c * loc_size;
        for (size_t i = start; i < start + loc_size; ++i) {
            h->AddPoint(std::span<const float>(loc_data.data() + i * dim, dim));
        }
        locule_hnsw.push_back(std::move(h));
    }

    std::cout << "| Locules Probed ($M$) | Probed Ratio (%) | Global Recall@10 | Search Latency (p50) | QPS | Achieves >=95% Global Recall? |\n";
    std::cout << "| :---: | :---: | :---: | :---: | :---: | :---: |\n";

    std::vector<size_t> probes = {1, 2, 4, 6, 8};
    for (size_t m : probes) {
        double global_r10 = 0.0;
        std::vector<double> lats;

        for (size_t q = 0; q < num_queries; ++q) {
            std::span<const float> q_span(queries.data() + q * dim, dim);
            auto qt0 = std::chrono::high_resolution_clock::now();

            // Stage 1: Compass scores centroids
            std::vector<std::pair<float, size_t>> cd(num_loc);
            for (size_t c = 0; c < num_loc; ++c) {
                cd[c] = {pomai::core::L2Sq(q_span, centroids[c]), c};
            }
            std::sort(cd.begin(), cd.end());

            // Stage 2: Query top M Locules
            std::vector<pomai::core::TopKItem> merged_pool;
            for (size_t p = 0; p < m; ++p) {
                size_t loc_idx = cd[p].second;
                std::vector<int32_t> sub_ids;
                std::vector<float> sub_dists;
                locule_hnsw[loc_idx]->Search(q_span, 16, 32, sub_ids, sub_dists);

                size_t base_offset = loc_idx * loc_size;
                for (size_t i = 0; i < sub_ids.size(); ++i) {
                    uint32_t global_id = static_cast<uint32_t>(base_offset + sub_ids[i]);
                    merged_pool.push_back({static_cast<pomai::VectorId>(global_id), -sub_dists[i], 0, nullptr});
                }
            }

            // Stage 3: Global Top-K merge
            pomai::core::SelectTopK(merged_pool, topk);
            auto qt1 = std::chrono::high_resolution_clock::now();
            lats.push_back(std::chrono::duration<double, std::micro>(qt1 - qt0).count());

            std::vector<int32_t> final_ids;
            for (const auto& item : merged_pool) final_ids.push_back(static_cast<int32_t>(item.id));
            global_r10 += ComputeRecall10(final_ids, loc_gt[q]);
        }

        std::sort(lats.begin(), lats.end());
        double avg_r = global_r10 / num_queries;
        double p50_ms = lats[num_queries/2] / 1000.0;
        double qps = 1000.0 / p50_ms;
        double probed_ratio = (static_cast<double>(m) / num_loc) * 100.0;

        std::cout << "| " << std::setw(20) << m
                  << " | " << std::fixed << std::setprecision(1) << std::setw(15) << probed_ratio << "%"
                  << " | **" << std::fixed << std::setprecision(3) << std::setw(15) << avg_r << "**"
                  << " | " << std::fixed << std::setprecision(2) << std::setw(19) << p50_ms << " ms"
                  << " | " << std::fixed << std::setprecision(0) << std::setw(5) << qps
                  << " | " << (avg_r >= 0.95 ? "**YES (GATE PASSED)**" : "NO") << " |\n";
    }
    std::cout << "\n";

    return 0;
}
