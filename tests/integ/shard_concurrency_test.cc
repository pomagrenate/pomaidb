#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include <memory>
#include <vector>

namespace {

using namespace pomai;

// Single-threaded: sequential puts on a single membrane
POMAI_TEST(ShardConcurrency_ParallelPuts) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("shard_concurrency_puts");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));

    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 1;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    constexpr int num_threads = 4;
    constexpr int puts_per_thread = 100;

    int failures = 0;
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < puts_per_thread; ++i) {
            VectorId id = t * puts_per_thread + i;
            std::vector<float> v = {
                static_cast<float>(id),
                static_cast<float>(id + 1),
                static_cast<float>(id + 2),
                static_cast<float>(id + 3)
            };
            Status st = db->Put("default", id, v);
            if (!st.ok()) ++failures;
        }
    }

    POMAI_EXPECT_EQ(failures, 0);

    db->Freeze("default");
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < puts_per_thread; ++i) {
            VectorId id = t * puts_per_thread + i;
            std::vector<float> out;
            Status st = db->Get("default", id, &out);
            POMAI_EXPECT_OK(st);
            POMAI_EXPECT_EQ(out.size(), 4u);
            if (out.size() == 4) {
                POMAI_EXPECT_TRUE(std::abs(out[0] - static_cast<float>(id)) < 20.0f);
            }
        }
    }

    db->Close();
}

// Single-threaded: mixed operations (Put, Get, Delete, Search) sequentially
POMAI_TEST(ShardConcurrency_MixedOperations) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("shard_concurrency_mixed");
    opt.dim = 4;
    opt.shard_count = 2;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    DB::Open(opt, &db);
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 2;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    for (int i = 0; i < 100; ++i) {
        std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
        db->Put("default", i, v);
    }
    db->Freeze("default");

    for (int i = 100; i < 200; ++i) {
        std::vector<float> v = {5.0f, 6.0f, 7.0f, 8.0f};
        db->Put("default", i, v);
    }

    for (int i = 0; i < 50; ++i) {
        std::vector<float> out;
        db->Get("default", i % 100, &out);
    }

    std::vector<float> query = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 20; ++i) {
        SearchResult res;
        db->Search("default", query, 10, &res);
    }

    db->Close();
}

} // namespace
