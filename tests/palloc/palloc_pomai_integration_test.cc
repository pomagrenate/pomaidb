#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "src/pomegranate_engine.h"
#include "src/search.h"
#include "src/types.h"
#include "src/utils/palloc_compat.h"

#include <vector>
#include <random>
#include <filesystem>
#include <iostream>

namespace pomai::palloc_qa {

POMAI_TEST(PallocIntegration_FullEngineMemoryChaos) {
    std::string db_dir = pomai::test::TempDir("palloc_engine_chaos");
    std::filesystem::create_directories(db_dir);

    constexpr uint32_t kDim = 64;
    constexpr size_t kNumVectors = 2000;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = kDim;
    opt.fsync = FsyncPolicy::kNever;

    pomai::core::PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_TRUE(engine.Open().ok());

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vec(kDim);

    // 1. Ingestion Storm: Ingest 2,000 vectors with frequent freezes/flushes
    for (uint64_t id = 1; id <= kNumVectors; ++id) {
        for (uint32_t d = 0; d < kDim; ++d) {
            vec[d] = dist(rng);
        }
        POMAI_EXPECT_TRUE(engine.Put(id, vec).ok());

        if (id % 500 == 0) {
            POMAI_EXPECT_TRUE(engine.Freeze().ok());
        }
    }

    // 2. Query Storm: Execute queries against live Rind and frozen FruitMap
    SearchResult results;
    for (uint64_t q = 0; q < 50; ++q) {
        for (uint32_t d = 0; d < kDim; ++d) {
            vec[d] = dist(rng);
        }
        results.Clear();
        POMAI_EXPECT_TRUE(engine.Search(vec, 10, &results).ok());
        POMAI_EXPECT_TRUE(!results.hits.empty());
    }

    // 3. Compaction / Press Storm: Press all segments into persistent locules
    POMAI_EXPECT_TRUE(engine.Compact().ok());

    // 4. Query after Press
    for (uint64_t q = 0; q < 50; ++q) {
        for (uint32_t d = 0; d < kDim; ++d) {
            vec[d] = dist(rng);
        }
        results.Clear();
        POMAI_EXPECT_TRUE(engine.Search(vec, 10, &results).ok());
        POMAI_EXPECT_TRUE(!results.hits.empty());
    }

    // 5. Deletions Storm: Delete 500 vectors
    for (uint64_t id = 1; id <= 500; ++id) {
        POMAI_EXPECT_TRUE(engine.Delete(id).ok());
    }

    // 6. Reopen Storm: Close engine and reopen from disk
    POMAI_EXPECT_TRUE(engine.Close().ok());

    pomai::core::PomegranateEngine reopened(opt, MetricType::kL2);
    POMAI_EXPECT_TRUE(reopened.Open().ok());

    // Verify deleted vectors do not appear in queries
    for (uint64_t q = 0; q < 50; ++q) {
        for (uint32_t d = 0; d < kDim; ++d) {
            vec[d] = dist(rng);
        }
        results.Clear();
        POMAI_EXPECT_TRUE(reopened.Search(vec, 10, &results).ok());
        for (const auto& hit : results.hits) {
            POMAI_EXPECT_TRUE(hit.id > 500); // Deleted IDs 1..500 must not appear
        }
    }

    POMAI_EXPECT_TRUE(reopened.Close().ok());
    std::error_code ec;
    std::filesystem::remove_all(db_dir, ec);
}

} // namespace pomai::palloc_qa
