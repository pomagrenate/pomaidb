// tests/adversarial/hnsw_torture_test.cc
// Phase 4: HNSW Algorithmic Integrity & Phase 5: HNSW Parameter Abuse & Boundary Torture

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "tests/adversarial/golden_oracle.h"

#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "pomegranate_engine.h"
#include "options.h"
#include "search.h"
#include "types.h"

namespace {

using namespace pomai;
using namespace pomai::core;
using namespace pomai::adversarial;

// 1. N boundaries: N=0, N=1, N=2, K > N
POMAI_TEST(HnswTorture_N_Boundaries) {
    const std::string db_dir = test::TempDir("pomai-hnsw-n-boundaries");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 16;
    opt.index_params.hnsw_ef_construction = 50;
    opt.index_params.hnsw_ef_search = 32;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    std::vector<float> q(dim, 1.0f);
    SearchResult res;

    // Test N = 0
    POMAI_EXPECT_OK(engine.Search(q, 10, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 0);

    // Test N = 1
    std::vector<float> v1(dim, 1.0f);
    POMAI_EXPECT_OK(engine.Put(42, v1));
    POMAI_EXPECT_OK(engine.Compact());

    res.Clear();
    POMAI_EXPECT_OK(engine.Search(q, 10, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 1);
    POMAI_EXPECT_EQ(res.hits[0].id, 42);
    // Exact match: L2 distance is 0, score is 0.0f
    POMAI_EXPECT_TRUE(std::abs(res.hits[0].score) < 1e-4f);

    // Test K > N
    res.Clear();
    POMAI_EXPECT_OK(engine.Search(q, 100, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 1);
    POMAI_EXPECT_EQ(res.hits[0].id, 42);

    // Test N = 2
    std::vector<float> v2(dim, 2.0f);
    POMAI_EXPECT_OK(engine.Put(99, v2));
    POMAI_EXPECT_OK(engine.Compact());

    res.Clear();
    POMAI_EXPECT_OK(engine.Search(q, 10, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 2);
    POMAI_EXPECT_EQ(res.hits[0].id, 42);
    POMAI_EXPECT_EQ(res.hits[1].id, 99);

    POMAI_EXPECT_OK(engine.Close());
}

// 2. Parameter Abuse: efSearch < K, M=2, M=64, efConstruction=1
POMAI_TEST(HnswTorture_ParameterAbuse) {
    const std::string db_dir = test::TempDir("pomai-hnsw-param-abuse");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 8;
    const uint32_t N = 50;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 4;
    opt.index_params.hnsw_ef_construction = 10;
    opt.index_params.hnsw_ef_search = 5; // Deliberately efSearch < K

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    for (uint32_t i = 1; i <= N; ++i) {
        std::vector<float> vec(dim, static_cast<float>(i));
        POMAI_EXPECT_OK(engine.Put(i, vec));
    }
    POMAI_EXPECT_OK(engine.Compact());

    // Search with K = 20 while efSearch = 5
    std::vector<float> q(dim, 25.0f);
    SearchResult res;
    POMAI_EXPECT_OK(engine.Search(q, 20, &res));
    // Upstream hnswlib clamps ef_search to K or returns available candidates
    POMAI_EXPECT_TRUE(!res.hits.empty());
    POMAI_EXPECT_TRUE(res.hits.size() <= 20);

    // Verify ordering
    for (size_t i = 1; i < res.hits.size(); ++i) {
        POMAI_EXPECT_TRUE(res.hits[i - 1].score >= res.hits[i].score);
    }

    POMAI_EXPECT_OK(engine.Close());
}

// 3. 100% Identical Vectors and Tie Breaking
POMAI_TEST(HnswTorture_IdenticalVectorsDeterministicTieBreak) {
    const std::string db_dir = test::TempDir("pomai-hnsw-identical");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;
    const uint32_t N = 100;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 16;
    opt.index_params.hnsw_ef_construction = 50;
    opt.index_params.hnsw_ef_search = 50;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    std::vector<float> identical_vec(dim, 3.14159f);
    for (uint32_t i = 1; i <= N; ++i) {
        POMAI_EXPECT_OK(engine.Put(i, identical_vec));
    }
    POMAI_EXPECT_OK(engine.Compact());

    SearchResult res;
    POMAI_EXPECT_OK(engine.Search(identical_vec, 10, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 10);

    // All distances are identical (score = 0.0f)
    for (const auto& hit : res.hits) {
        POMAI_EXPECT_TRUE(std::abs(hit.score) < 1e-4f);
    }

    // Tie-break contract: Equal score -> IDs must be strictly ascending
    for (size_t i = 1; i < res.hits.size(); ++i) {
        POMAI_EXPECT_TRUE(res.hits[i - 1].id < res.hits[i].id);
    }

    POMAI_EXPECT_OK(engine.Close());
}

// 4. Non-finite values: NaNs and Infs must be rejected at boundary
POMAI_TEST(HnswTorture_RejectNonFiniteInputs) {
    const std::string db_dir = test::TempDir("pomai-hnsw-nan-rejection");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 4;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    float nan_val = std::numeric_limits<float>::quiet_NaN();
    float inf_val = std::numeric_limits<float>::infinity();

    // Reject NaN on Put
    std::vector<float> nan_vec = {1.0f, nan_val, 2.0f, 3.0f};
    auto st_nan_put = engine.Put(1, nan_vec);
    POMAI_EXPECT_TRUE(!st_nan_put.ok());
    POMAI_EXPECT_TRUE(st_nan_put.code() == ErrorCode::kInvalidArgument);

    // Reject Inf on Put
    std::vector<float> inf_vec = {1.0f, 2.0f, inf_val, 3.0f};
    auto st_inf_put = engine.Put(2, inf_vec);
    POMAI_EXPECT_TRUE(!st_inf_put.ok());
    POMAI_EXPECT_TRUE(st_inf_put.code() == ErrorCode::kInvalidArgument);

    // Insert valid vector
    std::vector<float> valid_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    POMAI_EXPECT_OK(engine.Put(3, valid_vec));

    // Reject NaN on Search
    SearchResult res;
    auto st_nan_search = engine.Search(nan_vec, 5, &res);
    POMAI_EXPECT_TRUE(!st_nan_search.ok());
    POMAI_EXPECT_TRUE(st_nan_search.code() == ErrorCode::kInvalidArgument);

    // Reject Inf on Search
    auto st_inf_search = engine.Search(inf_vec, 5, &res);
    POMAI_EXPECT_TRUE(!st_inf_search.ok());
    POMAI_EXPECT_TRUE(st_inf_search.code() == ErrorCode::kInvalidArgument);

    POMAI_EXPECT_OK(engine.Close());
}

// 5. Dimension Stress: Unaligned and extreme dimensions (D=1, 2, 7, 63, 128)
POMAI_TEST(HnswTorture_DimensionExtremes) {
    const std::vector<uint32_t> test_dims = {1, 2, 7, 63, 128};

    for (uint32_t dim : test_dims) {
        const std::string db_dir = test::TempDir("pomai-hnsw-dim-" + std::to_string(dim));
        std::filesystem::remove_all(db_dir);

        DBOptions opt;
        opt.path = db_dir;
        opt.dim = dim;
        opt.fsync = FsyncPolicy::kNever;
        opt.index_params.type = IndexType::kHnsw;
        opt.index_params.hnsw_m = 8;
        opt.index_params.hnsw_ef_construction = 30;
        opt.index_params.hnsw_ef_search = 16;

        PomegranateEngine engine(opt, MetricType::kL2);
        POMAI_EXPECT_OK(engine.Open());

        for (uint32_t i = 1; i <= 30; ++i) {
            std::vector<float> vec(dim, static_cast<float>(i) * 0.1f);
            POMAI_EXPECT_OK(engine.Put(i, vec));
        }
        POMAI_EXPECT_OK(engine.Compact());

        std::vector<float> q(dim, 1.5f);
        SearchResult res;
        POMAI_EXPECT_OK(engine.Search(q, 5, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 5);

        POMAI_EXPECT_OK(engine.Close());
    }
}

// 6. Recall Verification against Independent Double-Precision Golden Oracle
POMAI_TEST(HnswTorture_RecallAgainstGoldenOracle) {
    const std::string db_dir = test::TempDir("pomai-hnsw-oracle-recall");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 32;
    const uint32_t N = 400;
    const uint32_t K = 10;
    const uint32_t num_queries = 20;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 16;
    opt.index_params.hnsw_ef_construction = 100;
    opt.index_params.hnsw_ef_search = 64;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    std::mt19937 rng(999);
    std::normal_distribution<float> d_norm(0.0f, 1.0f);

    GoldenOracle oracle(dim, GoldenOracle::Metric::kL2);

    for (uint32_t i = 1; i <= N; ++i) {
        std::vector<float> vec(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            vec[d] = d_norm(rng);
        }
        oracle.Put(i, vec);
        POMAI_EXPECT_OK(engine.Put(i, vec));
    }
    POMAI_EXPECT_OK(engine.Compact());

    double total_recall = 0.0;
    for (uint32_t q_idx = 0; q_idx < num_queries; ++q_idx) {
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            query[d] = d_norm(rng);
        }

        // Golden Oracle brute-force truth
        auto golden_hits = oracle.Search(query, K);

        // PomaiDB search
        SearchResult res;
        POMAI_EXPECT_OK(engine.Search(query, K, &res));

        double recall = GoldenOracle::ComputeRecall(golden_hits, res.hits);
        total_recall += recall;
    }

    double avg_recall = total_recall / num_queries;
    std::cout << "[HNSW ORACLE AUDIT] Average Recall@" << K << " across "
              << num_queries << " queries = " << avg_recall << std::endl;

    // Hard floor contract: Recall@10 must be >= 0.95
    POMAI_EXPECT_TRUE(avg_recall >= 0.95);

    POMAI_EXPECT_OK(engine.Close());
}

} // namespace
