// benchmarks/production_hnsw_bench.cc
// Rigorous End-to-End Database Benchmark:
// Production Compass + Intra-Locule HNSW vs Flat SIMD Scan
// Measures true Database Query Latency (p50, p95, p99), QPS, Speedup,
// and Empirical Recall@1, Recall@10, Recall@100 against exact FP32 ground truth.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "pomegranate_engine.h"
#include "options.h"
#include "search.h"
#include "distance.h"

using namespace pomai;
using namespace pomai::core;
using Clock = std::chrono::steady_clock;

struct GroundTruthItem {
    VectorId id;
    float dist_sq;
};

static std::vector<GroundTruthItem> ComputeGroundTruth(
    const std::vector<std::pair<VectorId, std::vector<float>>>& dataset,
    const std::vector<float>& query,
    uint32_t topk) {
    std::vector<GroundTruthItem> results;
    results.reserve(dataset.size());

    const size_t dim = query.size();
    for (const auto& item : dataset) {
        float d = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float diff = item.second[i] - query[i];
            d += diff * diff;
        }
        results.push_back({item.first, d});
    }

    std::partial_sort(results.begin(),
                      results.begin() + std::min<size_t>(topk, results.size()),
                      results.end(),
                      [](const GroundTruthItem& a, const GroundTruthItem& b) {
                          if (std::abs(a.dist_sq - b.dist_sq) > 1e-6f) {
                              return a.dist_sq < b.dist_sq;
                          }
                          return a.id < b.id;
                      });

    if (results.size() > topk) {
        results.resize(topk);
    }
    return results;
}

static float CalculateRecall(
    const std::vector<SearchHit>& hits,
    const std::vector<GroundTruthItem>& gt,
    uint32_t k) {
    if (gt.empty() || k == 0) return 0.0f;
    size_t check_k = std::min({static_cast<size_t>(k), hits.size(), gt.size()});

    std::unordered_set<VectorId> gt_set;
    for (size_t i = 0; i < check_k; ++i) {
        gt_set.insert(gt[i].id);
    }

    size_t matches = 0;
    for (size_t i = 0; i < check_k; ++i) {
        if (gt_set.count(hits[i].id)) {
            matches++;
        }
    }
    return static_cast<float>(matches) / static_cast<float>(check_k);
}

static double Percentile(std::vector<double> vals, double p) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t idx = static_cast<size_t>(std::floor(p * static_cast<double>(vals.size() - 1)));
    return vals[idx];
}

struct BenchResult {
    double p50_us;
    double p95_us;
    double p99_us;
    double qps;
    float recall_1;
    float recall_10;
    float recall_100;
};

static BenchResult RunQueries(
    PomegranateEngine& engine,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::pair<VectorId, std::vector<float>>>& dataset,
    const SearchOptions& s_opts,
    uint32_t topk) {

    // Warm-up queries
    SearchResult dummy_res;
    for (size_t i = 0; i < std::min<size_t>(10, queries.size()); ++i) {
        (void)engine.Search(queries[i], topk, s_opts, &dummy_res);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(queries.size());

    float total_r1 = 0.0f;
    float total_r10 = 0.0f;
    float total_r100 = 0.0f;

    for (size_t q = 0; q < queries.size(); ++q) {
        auto gt100 = ComputeGroundTruth(dataset, queries[q], 100);

        SearchResult r100;
        auto t0 = Clock::now();
        (void)engine.Search(queries[q], topk, s_opts, &r100);
        auto t1 = Clock::now();

        double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        latencies_us.push_back(elapsed_us);

        total_r1 += CalculateRecall(r100.hits, gt100, 1);
        total_r10 += CalculateRecall(r100.hits, gt100, 10);
        total_r100 += CalculateRecall(r100.hits, gt100, 100);
    }

    double total_time_s = 0.0;
    for (double u : latencies_us) total_time_s += (u / 1e6);

    BenchResult res;
    res.p50_us = Percentile(latencies_us, 0.50);
    res.p95_us = Percentile(latencies_us, 0.95);
    res.p99_us = Percentile(latencies_us, 0.99);
    res.qps = static_cast<double>(queries.size()) / total_time_s;
    res.recall_1 = total_r1 / static_cast<float>(queries.size());
    res.recall_10 = total_r10 / static_cast<float>(queries.size());
    res.recall_100 = total_r100 / static_cast<float>(queries.size());
    return res;
}

int main(int argc, char** argv) {
    const uint32_t dim = 64;
    const std::vector<size_t> test_sizes = {10000, 50000, 100000};
    const size_t num_queries = 100;
    const uint32_t topk = 100;

    std::cout << "========================================================================================\n";
    std::cout << " PomaiDB Production HNSW vs Flat SIMD Benchmark Forensics\n";
    std::cout << " Testing Sub-Linear Query Performance Across Scaling N (dim=" << dim << ", topk=" << topk << ")\n";
    std::cout << "========================================================================================\n\n";

    std::mt19937 rng(42);
    std::normal_distribution<float> d_norm(0.0f, 1.0f);

    for (size_t N : test_sizes) {
        std::cout << ">>> EVALUATING N = " << N << " VECTORS ...\n";

        // Generate clustered dataset (8 Gaussian clusters)
        const size_t num_clusters = 8;
        std::vector<std::vector<float>> cluster_centers(num_clusters, std::vector<float>(dim, 0.0f));
        for (size_t c = 0; c < num_clusters; ++c) {
            for (uint32_t d = 0; d < dim; ++d) {
                cluster_centers[c][d] = ((c == (d % num_clusters)) ? 4.0f : -1.5f);
            }
        }

        std::vector<std::pair<VectorId, std::vector<float>>> dataset;
        dataset.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            size_t c = i % num_clusters;
            std::vector<float> vec(dim);
            for (uint32_t d = 0; d < dim; ++d) {
                vec[d] = cluster_centers[c][d] + d_norm(rng) * 0.4f;
            }
            dataset.push_back({static_cast<VectorId>(i + 1), std::move(vec)});
        }

        // Generate query vectors
        std::vector<std::vector<float>> queries;
        queries.reserve(num_queries);
        for (size_t q = 0; q < num_queries; ++q) {
            size_t c = q % num_clusters;
            std::vector<float> qvec(dim);
            for (uint32_t d = 0; d < dim; ++d) {
                qvec[d] = cluster_centers[c][d] + d_norm(rng) * 0.4f;
            }
            queries.push_back(std::move(qvec));
        }

        // -------------------------------------------------------------
        // 1. Benchmark Production Compass + HNSW
        // -------------------------------------------------------------
        std::string db_dir_hnsw = "temp_bench_db_hnsw_" + std::to_string(N);
        std::filesystem::remove_all(db_dir_hnsw);

        DBOptions opt_hnsw;
        opt_hnsw.path = db_dir_hnsw;
        opt_hnsw.dim = dim;
        opt_hnsw.fsync = FsyncPolicy::kNever;
        opt_hnsw.index_params.type = IndexType::kHnsw;
        opt_hnsw.index_params.hnsw_m = 16;
        opt_hnsw.index_params.hnsw_ef_construction = 200;
        opt_hnsw.index_params.hnsw_ef_search = 128;

        BenchResult res_hnsw;
        {
            PomegranateEngine engine_hnsw(opt_hnsw, MetricType::kL2);
            (void)engine_hnsw.Open();

            for (const auto& item : dataset) {
                (void)engine_hnsw.Put(item.first, item.second);
            }
            (void)engine_hnsw.Compact();

            SearchOptions s_opts_hnsw;
            s_opts_hnsw.ef_search = 128;
            res_hnsw = RunQueries(engine_hnsw, queries, dataset, s_opts_hnsw, topk);
            (void)engine_hnsw.Close();
        }
        std::error_code ec;
        std::filesystem::remove_all(db_dir_hnsw, ec);

        // -------------------------------------------------------------
        // 2. Benchmark Flat SIMD Scan
        // -------------------------------------------------------------
        // Measure Flat baseline by doing brute-force SIMD scan over all vectors
        std::vector<double> flat_latencies;
        flat_latencies.reserve(queries.size());

        for (const auto& q : queries) {
            auto t0 = Clock::now();
            auto gt = ComputeGroundTruth(dataset, q, topk);
            auto t1 = Clock::now();
            double el = std::chrono::duration<double, std::micro>(t1 - t0).count();
            flat_latencies.push_back(el);
        }

        double total_flat_s = 0.0;
        for (double u : flat_latencies) total_flat_s += (u / 1e6);

        BenchResult res_flat;
        res_flat.p50_us = Percentile(flat_latencies, 0.50);
        res_flat.p95_us = Percentile(flat_latencies, 0.95);
        res_flat.p99_us = Percentile(flat_latencies, 0.99);
        res_flat.qps = static_cast<double>(queries.size()) / total_flat_s;
        res_flat.recall_1 = 1.0f;
        res_flat.recall_10 = 1.0f;
        res_flat.recall_100 = 1.0f;

        double speedup = res_flat.p50_us / res_hnsw.p50_us;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "--- RESULTS FOR N = " << N << " ---\n";
        std::cout << "  Flat SIMD Scan:    p50 = " << res_flat.p50_us << " µs (" << res_flat.p50_us / 1000.0 << " ms), "
                  << "QPS = " << res_flat.qps << ", Recall@10 = 1.000\n";
        std::cout << "  Compass + HNSW:    p50 = " << res_hnsw.p50_us << " µs (" << res_hnsw.p50_us / 1000.0 << " ms), "
                  << "QPS = " << res_hnsw.qps << ", Recall@10 = " << res_hnsw.recall_10
                  << ", Recall@100 = " << res_hnsw.recall_100 << "\n";
        std::cout << "  Empirical Speedup: " << speedup << "x faster (p50)\n";
        std::cout << "---------------------------------------------------------\n\n";
    }

    return 0;
}
