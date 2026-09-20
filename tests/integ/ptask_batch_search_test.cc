#include "tests/common/test_main.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>

#include "options.h"
#include "pomai.h"
#include "search.h"
#include "tests/common/test_tmpdir.h"
#include <ptask/ptask.h>

namespace {

static std::vector<float> MakeVec(std::uint32_t dim, float base) {
    std::vector<float> v(dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
        v[i] = base + static_cast<float>(i) * 0.01f;
    }
    return v;
}

POMAI_TEST(PTask_BatchSearch_AccuracyAndParity) {
    pomai::DBOptions opt;
    opt.path = pomai::test::TempDir("ptask-batch-accuracy");
    opt.dim = 16;
    opt.search_threads = 4; // Use 4 work-stealing threads from ptask
    opt.fsync = pomai::FsyncPolicy::kNever;

    std::unique_ptr<pomai::DB> db;
    POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

    pomai::MembraneSpec spec;
    spec.name = "default";
    spec.dim = opt.dim;
    POMAI_EXPECT_OK(db->CreateMembrane(spec));
    POMAI_EXPECT_OK(db->OpenMembrane("default"));

    // Ingest 500 vectors
    for (int i = 0; i < 500; ++i) {
        auto vec = MakeVec(opt.dim, static_cast<float>(i));
        POMAI_EXPECT_OK(db->Put(static_cast<pomai::VectorId>(100 + i), vec));
    }

    // Flush to Locule segments
    POMAI_EXPECT_OK(db->Flush());

    // Prepare 40 test queries
    constexpr uint32_t num_queries = 40;
    constexpr uint32_t topk = 5;
    std::vector<float> flat_queries;
    flat_queries.reserve(num_queries * opt.dim);

    std::vector<pomai::SearchResult> single_results(num_queries);

    for (uint32_t q = 0; q < num_queries; ++q) {
        auto qvec = MakeVec(opt.dim, static_cast<float>(q * 10 + 2));
        flat_queries.insert(flat_queries.end(), qvec.begin(), qvec.end());

        // Perform sequential individual Search
        POMAI_EXPECT_OK(db->Search(qvec, topk, &single_results[q]));
    }

    // Perform ptask-accelerated SearchBatch
    std::vector<pomai::SearchResult> batch_results;
    POMAI_EXPECT_OK(db->SearchBatch(flat_queries, num_queries, topk, &batch_results));

    POMAI_EXPECT_EQ(batch_results.size(), num_queries);

    // Verify 100% exact parity between sequential search and ptask work-stealing SearchBatch
    for (uint32_t q = 0; q < num_queries; ++q) {
        const auto& expected = single_results[q];
        const auto& actual = batch_results[q];

        POMAI_EXPECT_EQ(actual.hits.size(), expected.hits.size());
        for (size_t k = 0; k < expected.hits.size(); ++k) {
            POMAI_EXPECT_EQ(actual.hits[k].id, expected.hits[k].id);
            POMAI_EXPECT_TRUE(std::abs(actual.hits[k].score - expected.hits[k].score) < 1e-4f);
        }
    }
}

POMAI_TEST(PTask_Oversubscription_Clamping) {
    pomai::DBOptions opt;
    opt.path = pomai::test::TempDir("ptask-oversubscription");
    opt.dim = 8;
    // Extreme thread count: 5000 threads requested on user device
    opt.search_threads = 5000;
    opt.fsync = pomai::FsyncPolicy::kNever;

    std::unique_ptr<pomai::DB> db;
    POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

    pomai::MembraneSpec spec;
    spec.name = "default";
    spec.dim = opt.dim;
    POMAI_EXPECT_OK(db->CreateMembrane(spec));
    POMAI_EXPECT_OK(db->OpenMembrane("default"));

    // Insert 100 vectors
    for (int i = 0; i < 100; ++i) {
        auto vec = MakeVec(opt.dim, static_cast<float>(i));
        POMAI_EXPECT_OK(db->Put(static_cast<pomai::VectorId>(i + 1), vec));
    }

    // Verify ptask clamped worker threads safely and executed batch without error or crash
    constexpr uint32_t num_queries = 20;
    std::vector<float> flat_queries;
    for (uint32_t q = 0; q < num_queries; ++q) {
        auto qvec = MakeVec(opt.dim, static_cast<float>(q * 3));
        flat_queries.insert(flat_queries.end(), qvec.begin(), qvec.end());
    }

    std::vector<pomai::SearchResult> batch_results;
    POMAI_EXPECT_OK(db->SearchBatch(flat_queries, num_queries, 3, &batch_results));
    POMAI_EXPECT_EQ(batch_results.size(), num_queries);
    for (const auto& res : batch_results) {
        POMAI_EXPECT_EQ(res.hits.size(), 3u);
    }
}

} // namespace
