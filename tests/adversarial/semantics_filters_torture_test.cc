// tests/adversarial/semantics_filters_torture_test.cc
// Phase 12: Metadata Filter Torture, Phase 13: Metric Semantics,
// Phase 15: Dimension Mismatches, Phase 16: Top-K Extremes

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>

#include "pomegranate_engine.h"
#include "options.h"
#include "search.h"
#include "types.h"
#include "metadata.h"

namespace {

using namespace pomai;
using namespace pomai::core;

// 1. Phase 12: Metadata Filter Selectivity Extremes (0%, 0.5%, 100%)
POMAI_TEST(Filters_SelectivityExtremes) {
    const std::string db_dir = test::TempDir("pomai-filters-selectivity");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 8;
    const uint32_t N = 200;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    // Insert 199 common vectors and 1 rare target vector
    for (uint32_t i = 1; i <= N; ++i) {
        std::vector<float> vec(dim, static_cast<float>(i));
        Metadata meta;
        if (i == 42) {
            meta.device_id = "target_rare_device";
            meta.location_id = "secret_lab";
        } else {
            meta.device_id = "common_device";
            meta.location_id = "warehouse";
        }
        POMAI_EXPECT_OK(engine.Put(i, vec, meta));
    }
    POMAI_EXPECT_OK(engine.Compact());

    std::vector<float> q(dim, 42.0f);

    // Case A: Selectivity 0% (filter matches no records)
    {
        SearchOptions sopts;
        sopts.filters.push_back(Filter("device_id", "does_not_exist"));
        SearchResult res;
        POMAI_EXPECT_OK(engine.Search(q, 10, sopts, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 0);
    }

    // Case B: Selectivity 0.5% (filter matches exactly 1 out of 200 records)
    {
        SearchOptions sopts;
        sopts.filters.push_back(Filter("device_id", "target_rare_device"));
        SearchResult res;
        POMAI_EXPECT_OK(engine.Search(q, 10, sopts, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 1);
        POMAI_EXPECT_EQ(res.hits[0].id, 42);
    }

    // Case C: Selectivity 99.5% (filter matches common records)
    {
        SearchOptions sopts;
        sopts.filters.push_back(Filter("device_id", "common_device"));
        SearchResult res;
        POMAI_EXPECT_OK(engine.Search(q, 10, sopts, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 10);
        // Ensure the rare vector 42 is excluded
        for (const auto& h : res.hits) {
            POMAI_EXPECT_TRUE(h.id != 42);
        }
    }

    POMAI_EXPECT_OK(engine.Close());
}

// 2. Phase 15: Dimension Mismatches
POMAI_TEST(Dimensions_StrictMismatchRejection) {
    const std::string db_dir = test::TempDir("pomai-dim-mismatch");
    std::filesystem::remove_all(db_dir);
    const uint32_t declared_dim = 16;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = declared_dim;
    opt.fsync = FsyncPolicy::kNever;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    std::vector<float> too_small(8, 1.0f);
    std::vector<float> too_large(32, 1.0f);
    std::vector<float> correct(16, 1.0f);

    // Rejection on Put
    auto st_small_put = engine.Put(1, too_small);
    POMAI_EXPECT_TRUE(!st_small_put.ok());
    POMAI_EXPECT_TRUE(st_small_put.code() == ErrorCode::kInvalidArgument);

    auto st_large_put = engine.Put(2, too_large);
    POMAI_EXPECT_TRUE(!st_large_put.ok());
    POMAI_EXPECT_TRUE(st_large_put.code() == ErrorCode::kInvalidArgument);

    POMAI_EXPECT_OK(engine.Put(3, correct));

    // Rejection on Search
    SearchResult res;
    auto st_small_search = engine.Search(too_small, 5, &res);
    POMAI_EXPECT_TRUE(!st_small_search.ok());
    POMAI_EXPECT_TRUE(st_small_search.code() == ErrorCode::kInvalidArgument);

    auto st_large_search = engine.Search(too_large, 5, &res);
    POMAI_EXPECT_TRUE(!st_large_search.ok());
    POMAI_EXPECT_TRUE(st_large_search.code() == ErrorCode::kInvalidArgument);

    POMAI_EXPECT_OK(engine.Close());
}

// 3. Phase 16: Top-K Boundary and Extremes (K=0, K=1, K > N)
POMAI_TEST(TopK_Extremes_SafeExecution) {
    const std::string db_dir = test::TempDir("pomai-topk-extremes");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 8;
    const uint32_t N = 10;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    for (uint32_t i = 1; i <= N; ++i) {
        std::vector<float> vec(dim, static_cast<float>(i));
        POMAI_EXPECT_OK(engine.Put(i, vec));
    }
    POMAI_EXPECT_OK(engine.Compact());

    std::vector<float> q(dim, 3.0f);

    // K = 0: Must return 0 hits, Status::Ok()
    {
        SearchResult res;
        POMAI_EXPECT_OK(engine.Search(q, 0, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 0);
    }

    // K = 1: Must return exactly 1 hit (ID 3)
    {
        SearchResult res;
        POMAI_EXPECT_OK(engine.Search(q, 1, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 1);
        POMAI_EXPECT_EQ(res.hits[0].id, 3);
        POMAI_EXPECT_TRUE(std::abs(res.hits[0].score) < 1e-4f);
    }

    // K = 50,000 (vastly exceeding N = 10): Must return all 10 hits without buffer overflow
    {
        SearchResult res;
        POMAI_EXPECT_OK(engine.Search(q, 50000, &res));
        POMAI_EXPECT_EQ(res.hits.size(), N);
        POMAI_EXPECT_EQ(res.hits[0].id, 3);
    }

    POMAI_EXPECT_OK(engine.Close());
}

} // namespace
