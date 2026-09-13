// crossover_bench.cc — ANN vs Compass vs Flat Crossover Evaluation
//
// Evaluates:
// 1. Flat SIMD Brute Force
// 2. Compass Routing + Flat Scan
// 3. HNSW Graph Traversal
// Across N = 1000, 5000, 10000, 25000 vectors.
// Measures Latency (p50, p95, p99), QPS, and Recall@1, Recall@10, Recall@100.
//
// Copyright 2026 PomaiDB authors. MIT License.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

#include "distance.h"
#include "topk.h"
#include "hnsw_index.h"

namespace {

using Clock = std::chrono::steady_clock;

std::vector<float> MakeRandomVec(size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (size_t i = 0; i < dim; ++i) v[i] = dist(rng);
    return v;
}

double Percentile(std::vector<double> vals, double p) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t idx = static_cast<size_t>(std::floor(p * static_cast<double>(vals.size() - 1)));
    return vals[idx];
}

double ComputeRecall(const std::vector<pomai::VectorId>& ground_truth,
                     const std::vector<pomai::VectorId>& candidate,
                     size_t k) {
    if (k == 0 || ground_truth.empty() || candidate.empty()) return 0.0;
    size_t limit = std::min({k, ground_truth.size(), candidate.size()});
    std::unordered_set<pomai::VectorId> gt_set(ground_truth.begin(), ground_truth.begin() + limit);
    size_t matches = 0;
    for (size_t i = 0; i < limit; ++i) {
        if (gt_set.count(candidate[i])) {
            ++matches;
        }
    }
    return static_cast<double>(matches) / static_cast<double>(limit);
}

struct EvalMetrics {
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double qps = 0.0;
    double recall_1 = 0.0;
    double recall_10 = 0.0;
    double recall_100 = 0.0;
};

} // namespace

int main() {
    pomai::core::InitDistance();

    const uint32_t dim = 64;
    const std::vector<size_t> n_sizes = {1000, 5000, 10000, 25000};
    const size_t num_queries = 200;

    std::cout << "========================================================================================\n";
    std::cout << " POMAI DB — ANN vs COMPASS vs FLAT SCAN CROSSOVER EVALUATION\n";
    std::cout << " Dimension: " << dim << ", Queries: " << num_queries << "\n";
    std::cout << "========================================================================================\n\n";

    // Pre-generate queries
    std::vector<std::vector<float>> queries(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
        queries[q] = MakeRandomVec(dim, static_cast<uint32_t>(10000 + q));
    }

    for (size_t N : n_sizes) {
        std::cout << ">>> DATASET SIZE N = " << N << " <<<\n";

        // Generate database vectors
        std::vector<float> db_vectors(N * dim);
        for (size_t i = 0; i < N; ++i) {
            auto v = MakeRandomVec(dim, static_cast<uint32_t>(i + 1));
            std::copy(v.begin(), v.end(), db_vectors.begin() + i * dim);
        }

        // Build HNSW index
        pomai::index::HnswOptions hnsw_opts;
        hnsw_opts.M = 16;
        hnsw_opts.ef_construction = 100;
        hnsw_opts.ef_search = 32;
        pomai::index::HnswIndex hnsw(dim, hnsw_opts, pomai::MetricType::kL2);

        for (size_t i = 0; i < N; ++i) {
            std::span<const float> v(db_vectors.data() + i * dim, dim);
            hnsw.Add(static_cast<pomai::VectorId>(i + 1), v);
        }

        // Partition dataset into simulated Compass Locules (e.g. 8 clusters of N/8 vectors)
        const size_t num_clusters = std::max<size_t>(1, N / 500);
        struct Cluster {
            std::vector<float> centroid;
            std::vector<size_t> indices;
        };
        std::vector<Cluster> clusters(num_clusters);
        for (size_t c = 0; c < num_clusters; ++c) {
            clusters[c].centroid = MakeRandomVec(dim, static_cast<uint32_t>(90000 + c));
        }
        for (size_t i = 0; i < N; ++i) {
            clusters[i % num_clusters].indices.push_back(i);
        }

        // 1. Ground truth exact Flat Brute Force
        std::vector<std::vector<pomai::VectorId>> ground_truth(num_queries);
        std::vector<double> flat_lats;
        flat_lats.reserve(num_queries);

        for (size_t q = 0; q < num_queries; ++q) {
            std::span<const float> q_span(queries[q]);
            const auto t0 = Clock::now();

            std::vector<pomai::core::TopKItem> items(N);
            for (size_t i = 0; i < N; ++i) {
                std::span<const float> v(db_vectors.data() + i * dim, dim);
                items[i].id = static_cast<pomai::VectorId>(i + 1);
                items[i].score = -pomai::core::L2Sq(q_span, v);
            }
            pomai::core::SelectTopK(items, 100);

            const auto t1 = Clock::now();
            flat_lats.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());

            ground_truth[q].resize(items.size());
            for (size_t k = 0; k < items.size(); ++k) ground_truth[q][k] = items[k].id;
        }

        EvalMetrics flat_m;
        flat_m.p50_us = Percentile(flat_lats, 0.50);
        flat_m.p95_us = Percentile(flat_lats, 0.95);
        flat_m.p99_us = Percentile(flat_lats, 0.99);
        double total_flat_sec = 0.0;
        for (double l : flat_lats) total_flat_sec += (l * 1e-6);
        flat_m.qps = total_flat_sec > 0.0 ? (static_cast<double>(num_queries) / total_flat_sec) : 0.0;
        flat_m.recall_1 = 1.0;
        flat_m.recall_10 = 1.0;
        flat_m.recall_100 = 1.0;

        // 2. Compass Routing + Flat Scan (probe top 2 clusters)
        std::vector<double> compass_lats;
        compass_lats.reserve(num_queries);
        double compass_r1 = 0.0, compass_r10 = 0.0, compass_r100 = 0.0;

        for (size_t q = 0; q < num_queries; ++q) {
            std::span<const float> q_span(queries[q]);
            const auto t0 = Clock::now();

            // Orient: score centroids
            std::vector<std::pair<float, size_t>> centroid_scores(num_clusters);
            for (size_t c = 0; c < num_clusters; ++c) {
                centroid_scores[c] = {pomai::core::L2Sq(q_span, clusters[c].centroid), c};
            }
            std::sort(centroid_scores.begin(), centroid_scores.end());

            // Scan probed clusters
            size_t probe_count = std::min<size_t>(num_clusters, 2);
            std::vector<pomai::core::TopKItem> items;
            for (size_t p = 0; p < probe_count; ++p) {
                size_t c_idx = centroid_scores[p].second;
                for (size_t idx : clusters[c_idx].indices) {
                    std::span<const float> v(db_vectors.data() + idx * dim, dim);
                    items.push_back({static_cast<pomai::VectorId>(idx + 1), -pomai::core::L2Sq(q_span, v), 0, nullptr});
                }
            }
            pomai::core::SelectTopK(items, 100);

            const auto t1 = Clock::now();
            compass_lats.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());

            std::vector<pomai::VectorId> cand_ids(items.size());
            for (size_t k = 0; k < items.size(); ++k) cand_ids[k] = items[k].id;

            compass_r1 += ComputeRecall(ground_truth[q], cand_ids, 1);
            compass_r10 += ComputeRecall(ground_truth[q], cand_ids, 10);
            compass_r100 += ComputeRecall(ground_truth[q], cand_ids, 100);
        }

        EvalMetrics compass_m;
        compass_m.p50_us = Percentile(compass_lats, 0.50);
        compass_m.p95_us = Percentile(compass_lats, 0.95);
        compass_m.p99_us = Percentile(compass_lats, 0.99);
        double total_compass_sec = 0.0;
        for (double l : compass_lats) total_compass_sec += (l * 1e-6);
        compass_m.qps = total_compass_sec > 0.0 ? (static_cast<double>(num_queries) / total_compass_sec) : 0.0;
        compass_m.recall_1 = compass_r1 / num_queries;
        compass_m.recall_10 = compass_r10 / num_queries;
        compass_m.recall_100 = compass_r100 / num_queries;

        // 3. HNSW Search
        std::vector<double> hnsw_lats;
        hnsw_lats.reserve(num_queries);
        double hnsw_r1 = 0.0, hnsw_r10 = 0.0, hnsw_r100 = 0.0;

        for (size_t q = 0; q < num_queries; ++q) {
            std::span<const float> q_span(queries[q]);
            const auto t0 = Clock::now();

            std::vector<pomai::VectorId> out_ids;
            std::vector<float> out_dists;
            hnsw.Search(q_span, 100, 32, &out_ids, &out_dists);

            const auto t1 = Clock::now();
            hnsw_lats.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());

            hnsw_r1 += ComputeRecall(ground_truth[q], out_ids, 1);
            hnsw_r10 += ComputeRecall(ground_truth[q], out_ids, 10);
            hnsw_r100 += ComputeRecall(ground_truth[q], out_ids, 100);
        }

        EvalMetrics hnsw_m;
        hnsw_m.p50_us = Percentile(hnsw_lats, 0.50);
        hnsw_m.p95_us = Percentile(hnsw_lats, 0.95);
        hnsw_m.p99_us = Percentile(hnsw_lats, 0.99);
        double total_hnsw_sec = 0.0;
        for (double l : hnsw_lats) total_hnsw_sec += (l * 1e-6);
        hnsw_m.qps = total_hnsw_sec > 0.0 ? (static_cast<double>(num_queries) / total_hnsw_sec) : 0.0;
        hnsw_m.recall_1 = hnsw_r1 / num_queries;
        hnsw_m.recall_10 = hnsw_r10 / num_queries;
        hnsw_m.recall_100 = hnsw_r100 / num_queries;

        // Output table
        std::cout << std::left << std::setw(22) << "Configuration"
                  << std::setw(12) << "p50 (us)"
                  << std::setw(12) << "p95 (us)"
                  << std::setw(12) << "p99 (us)"
                  << std::setw(10) << "QPS"
                  << std::setw(12) << "Recall@1"
                  << std::setw(12) << "Recall@10"
                  << std::setw(12) << "Recall@100" << "\n";
        std::cout << std::string(104, '-') << "\n";

        auto print_row = [](const std::string& name, const EvalMetrics& m) {
            std::cout << std::left << std::setw(22) << name
                      << std::setw(12) << std::fixed << std::setprecision(1) << m.p50_us
                      << std::setw(12) << std::fixed << std::setprecision(1) << m.p95_us
                      << std::setw(12) << std::fixed << std::setprecision(1) << m.p99_us
                      << std::setw(10) << std::fixed << std::setprecision(0) << m.qps
                      << std::setw(12) << std::fixed << std::setprecision(3) << m.recall_1
                      << std::setw(12) << std::fixed << std::setprecision(3) << m.recall_10
                      << std::setw(12) << std::fixed << std::setprecision(3) << m.recall_100 << "\n";
        };

        print_row("Flat SIMD (Brute)", flat_m);
        print_row("Compass + Flat", compass_m);
        print_row("HNSW Graph", hnsw_m);
        std::cout << "\n";
    }

    return 0;
}
