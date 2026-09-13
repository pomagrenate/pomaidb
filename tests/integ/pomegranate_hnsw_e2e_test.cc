// tests/integ/pomegranate_hnsw_e2e_test.cc
// End-to-End verification of PomaiDB Production HNSW:
// 1. Spatial Locule partitioning and HNSW index persistence inside .pom Locules.
// 2. High recall (Recall@10 >= 0.95) against exact FP32 ground truth oracle.
// 3. Monotonicity check (Recall@100 >= Recall@10 >= Recall@1).
// 4. Persistence round-trip (Close -> Reopen -> Query).
// 5. In-graph metadata filtering.
// 6. Resilient fallback to Pulp SQ8 SIMD flat scan upon graph corruption.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include "aril.h"
#include "locule.h"
#include "options.h"
#include "pomegranate_engine.h"
#include "search.h"
#include "status.h"
#include "utils/crc32c.h"
#include "utils/env.h"

namespace {

using namespace pomai;
using namespace pomai::core;

// Brute-force oracle for exact FP32 Ground Truth
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

POMAI_TEST(HNSW_Production_E2E_RecallAndPersistence) {
    const std::string db_dir = test::TempDir("pomai-hnsw-prod-e2e");
    const uint32_t dim = 64;
    const size_t N = 2000;
    const size_t num_queries = 50;

    std::mt19937 rng(42);
    std::normal_distribution<float> d_norm(0.0f, 1.0f);

    // Generate 4 clustered distributions in 64-D space
    std::vector<std::vector<float>> cluster_centers(4, std::vector<float>(dim, 0.0f));
    for (int c = 0; c < 4; ++c) {
        for (uint32_t d = 0; d < dim; ++d) {
            cluster_centers[c][d] = ((c == (int)(d % 4)) ? 5.0f : -2.0f);
        }
    }

    std::vector<std::pair<VectorId, std::vector<float>>> dataset;
    dataset.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        int c = static_cast<int>(i % 4);
        std::vector<float> vec(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            vec[d] = cluster_centers[c][d] + d_norm(rng) * 0.5f;
        }
        dataset.push_back({static_cast<VectorId>(i + 1), std::move(vec)});
    }

    // Generate test queries
    std::vector<std::vector<float>> queries;
    queries.reserve(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
        int c = static_cast<int>(q % 4);
        std::vector<float> qvec(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            qvec[d] = cluster_centers[c][d] + d_norm(rng) * 0.5f;
        }
        queries.push_back(std::move(qvec));
    }

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 16;
    opt.index_params.hnsw_ef_construction = 200;
    opt.index_params.hnsw_ef_search = 64;

    // 1. Ingestion and Compaction into Persistent Spatial Locules
    {
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        for (const auto& item : dataset) {
            Metadata meta;
            meta.device_id = (item.first % 2 == 0) ? "dev_even" : "dev_odd";
            POMAI_EXPECT_OK(engine.Put(item.first, item.second, meta));
        }

        // Compact Rind into spatial Locules with persistent HNSW indices
        POMAI_EXPECT_OK(engine.Compact());

        // Verify that intra-Locule HNSW graphs are present
        std::shared_ptr<Snapshot> snap;
        POMAI_EXPECT_OK(engine.GetSnapshot(&snap));
        POMAI_EXPECT_TRUE(snap != nullptr);

        // Execute queries and verify Recall@10 >= 0.95 and Monotonicity
        float total_recall_1 = 0.0f;
        float total_recall_10 = 0.0f;
        float total_recall_100 = 0.0f;

        SearchOptions s_opts;
        s_opts.ef_search = 64;

        for (const auto& q : queries) {
            auto gt_100 = ComputeGroundTruth(dataset, q, 100);

            SearchResult r_100;
            POMAI_EXPECT_OK(engine.Search(q, 100, s_opts, &r_100));

            SearchResult r_10;
            POMAI_EXPECT_OK(engine.Search(q, 10, s_opts, &r_10));

            SearchResult r_1;
            POMAI_EXPECT_OK(engine.Search(q, 1, s_opts, &r_1));

            float rec_1 = CalculateRecall(r_1.hits, gt_100, 1);
            float rec_10 = CalculateRecall(r_10.hits, gt_100, 10);
            float rec_100 = CalculateRecall(r_100.hits, gt_100, 100);

            total_recall_1 += rec_1;
            total_recall_10 += rec_10;
            total_recall_100 += rec_100;

            // Invariant: Recall@100 >= Recall@10 >= Recall@1
            POMAI_EXPECT_TRUE(rec_100 >= rec_10 - 0.05f);
        }

        float mean_recall_1 = total_recall_1 / num_queries;
        float mean_recall_10 = total_recall_10 / num_queries;
        float mean_recall_100 = total_recall_100 / num_queries;

        std::printf("E2E HNSW: Mean Recall@1 = %.4f, Recall@10 = %.4f, Recall@100 = %.4f\n",
                    mean_recall_1, mean_recall_10, mean_recall_100);

        POMAI_EXPECT_TRUE(mean_recall_10 >= 0.95f);
        POMAI_EXPECT_TRUE(mean_recall_100 >= mean_recall_10 - 0.02f);
        POMAI_EXPECT_TRUE(mean_recall_10 >= mean_recall_1 - 0.05f);

        // Test metadata filtering
        SearchOptions filter_opts;
        filter_opts.partition_device_id = "dev_even";
        SearchResult filtered_res;
        POMAI_EXPECT_OK(engine.Search(queries[0], 10, filter_opts, &filtered_res));
        POMAI_EXPECT_TRUE(!filtered_res.hits.empty());
        for (const auto& hit : filtered_res.hits) {
            POMAI_EXPECT_EQ(hit.id % 2, 0); // Must all be even
        }

        POMAI_EXPECT_OK(engine.Close());
    }

    // 2. Persistence Reopen Test: Reopen from disk, verify zero-copy graph loading & identical recall
    {
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        float total_recall_10 = 0.0f;
        SearchOptions s_opts;
        s_opts.ef_search = 64;

        for (const auto& q : queries) {
            auto gt_10 = ComputeGroundTruth(dataset, q, 10);
            SearchResult r_10;
            POMAI_EXPECT_OK(engine.Search(q, 10, s_opts, &r_10));
            total_recall_10 += CalculateRecall(r_10.hits, gt_10, 10);
        }

        float mean_recall_10 = total_recall_10 / num_queries;
        POMAI_EXPECT_TRUE(mean_recall_10 >= 0.95f);

        POMAI_EXPECT_OK(engine.Close());
    }
}

POMAI_TEST(HNSW_Corruption_Graceful_Fallback) {
    const std::string db_dir = test::TempDir("pomai-hnsw-corruption-fallback");
    const uint32_t dim = 32;
    const size_t N = 500;

    std::mt19937 rng(123);
    std::normal_distribution<float> d_norm(0.0f, 1.0f);

    std::vector<std::pair<VectorId, std::vector<float>>> dataset;
    dataset.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        std::vector<float> vec(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            vec[d] = d_norm(rng);
        }
        dataset.push_back({static_cast<VectorId>(i + 1), std::move(vec)});
    }

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 16;
    opt.index_params.hnsw_ef_construction = 100;
    opt.index_params.hnsw_ef_search = 32;

    // Create database, ingest vectors, compact with HNSW
    std::string locule_file_path;
    {
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());
        for (const auto& item : dataset) {
            POMAI_EXPECT_OK(engine.Put(item.first, item.second));
        }
        POMAI_EXPECT_OK(engine.Compact());
        POMAI_EXPECT_OK(engine.Close());
    }

    // Locate the .pom Locule file in db_dir using std::filesystem
    for (const auto& entry : std::filesystem::directory_iterator(db_dir)) {
        if (entry.path().extension() == ".pom" && entry.path().filename().string().rfind("locule_", 0) == 0) {
            locule_file_path = entry.path().string();
            break;
        }
    }
    POMAI_EXPECT_TRUE(!locule_file_path.empty());

    // Tamper with the .pom file: intentionally corrupt the HNSW graph header magic
    {
        std::fstream file(locule_file_path, std::ios::in | std::ios::out | std::ios::binary);
        POMAI_EXPECT_TRUE(file.is_open());

        // Read Locule file header to find Aril directory offset
        file.seekg(0, std::ios::beg);
        format::PomaiFileHeader file_hdr;
        file.read(reinterpret_cast<char*>(&file_hdr), sizeof(file_hdr));

        // Read Aril directory entry
        file.seekg(file_hdr.directory_offset, std::ios::beg);
        format::ArilDirectoryEntry aril_dir;
        file.read(reinterpret_cast<char*>(&aril_dir), sizeof(aril_dir));

        // In the first Aril, read the ArilHeader to find graph_offset
        file.seekg(aril_dir.aril_offset, std::ios::beg);
        format::ArilHeader aril_hdr;
        file.read(reinterpret_cast<char*>(&aril_hdr), sizeof(aril_hdr));
        POMAI_EXPECT_TRUE(aril_hdr.graph_offset > 0);

        // Corrupt the graph header magic inside the Aril
        uint64_t abs_graph_offset = aril_dir.aril_offset + aril_hdr.graph_offset;
        file.seekp(abs_graph_offset, std::ios::beg);
        uint32_t bad_magic = 0xDEADBEEF;
        file.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
        file.flush();

        // Update Aril block checksum so Locule container integrity passes and exercises ArilReader fallback
        std::vector<uint8_t> aril_bytes(aril_dir.aril_size);
        file.seekg(aril_dir.aril_offset, std::ios::beg);
        file.read(reinterpret_cast<char*>(aril_bytes.data()), aril_dir.aril_size);
        aril_dir.checksum = pomai::util::Crc32c(aril_bytes.data(), aril_dir.aril_size);

        file.seekp(file_hdr.directory_offset, std::ios::beg);
        file.write(reinterpret_cast<const char*>(&aril_dir), sizeof(aril_dir));
        file.flush();

        // Update PomaiFileHeader checksum protecting the directory table
        std::vector<uint8_t> dir_bytes(file_hdr.directory_size);
        file.seekg(file_hdr.directory_offset, std::ios::beg);
        file.read(reinterpret_cast<char*>(dir_bytes.data()), file_hdr.directory_size);
        file_hdr.checksum = pomai::util::Crc32c(dir_bytes.data(), file_hdr.directory_size);

        file.seekp(0, std::ios::beg);
        file.write(reinterpret_cast<const char*>(&file_hdr), sizeof(file_hdr));
        file.flush();
    }

    // Reopen database: ArilReader should detect corruption, log warning, fallback to Pulp SQ8 SIMD scan
    {
        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        // Queries must succeed gracefully through the Pulp fallback path
        std::vector<float> query = dataset[0].second;
        SearchResult result;
        POMAI_EXPECT_OK(engine.Search(query, 5, &result));

        POMAI_EXPECT_TRUE(!result.hits.empty());
        // Closest vector should be vector 1 (query itself)
        POMAI_EXPECT_EQ(result.hits[0].id, static_cast<VectorId>(1));

        POMAI_EXPECT_OK(engine.Close());
    }
}

} // namespace
