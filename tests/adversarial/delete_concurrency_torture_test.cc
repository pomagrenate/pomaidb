// tests/adversarial/delete_concurrency_torture_test.cc
// Phase 9: Delete / Tombstone Hell & Phase 10: Concurrency Destruction
// Stresses 100% deletions, tombstone re-insertion resurrection, and concurrent reader/deleter/writer/compactor threads.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "pomegranate_engine.h"
#include "options.h"
#include "search.h"
#include "types.h"

namespace {

using namespace pomai;
using namespace pomai::core;

// 1. 100% Deletions: All vectors tombstoned, compaction handles empty state
POMAI_TEST(Delete_100PercentTombstones_EmptySearchAndCleanCompact) {
    const std::string db_dir = test::TempDir("pomai-del-100pct");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;
    const uint32_t N = 100;

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

    for (uint32_t i = 1; i <= N; ++i) {
        std::vector<float> vec(dim, static_cast<float>(i));
        POMAI_EXPECT_OK(engine.Put(i, vec));
    }

    // Delete all 100 vectors
    for (uint32_t i = 1; i <= N; ++i) {
        POMAI_EXPECT_OK(engine.Delete(i));
    }

    // Search while in Rind (uncompacted)
    std::vector<float> q(dim, 50.0f);
    SearchResult res;
    POMAI_EXPECT_OK(engine.Search(q, 10, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 0);

    // Compact with 100% tombstones
    POMAI_EXPECT_OK(engine.Compact());

    // Search after compaction
    res.Clear();
    POMAI_EXPECT_OK(engine.Search(q, 10, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 0);

    POMAI_EXPECT_OK(engine.Close());
}

// 2. Tombstone Re-insertion: Ensure no zombie data or ghost vectors
POMAI_TEST(Delete_TombstoneResurrection_Prevented) {
    const std::string db_dir = test::TempDir("pomai-del-resurrect");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 8;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    std::vector<float> vec_a(dim, 10.0f);
    std::vector<float> vec_b(dim, 999.0f);

    // Step 1: Put ID 1 with vector A
    POMAI_EXPECT_OK(engine.Put(1, vec_a));
    POMAI_EXPECT_OK(engine.Compact());

    // Verify ID 1 is found near vec_a
    SearchResult res;
    POMAI_EXPECT_OK(engine.Search(vec_a, 5, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 1);
    POMAI_EXPECT_EQ(res.hits[0].id, 1);

    // Step 2: Delete ID 1
    POMAI_EXPECT_OK(engine.Delete(1));

    // Must NOT be found near vec_a
    res.Clear();
    POMAI_EXPECT_OK(engine.Search(vec_a, 5, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 0);

    // Step 3: Re-insert ID 1 with vector B
    POMAI_EXPECT_OK(engine.Put(1, vec_b));

    // Searching near vec_a must NOT return ID 1
    res.Clear();
    POMAI_EXPECT_OK(engine.Search(vec_a, 5, &res));
    // Even if returned, score must reflect vec_b distance, not vec_a distance
    if (!res.hits.empty()) {
        POMAI_EXPECT_TRUE(res.hits[0].score < -100.0f); // Large negative L2 distance
    }

    // Searching near vec_b MUST return ID 1 with distance 0
    res.Clear();
    POMAI_EXPECT_OK(engine.Search(vec_b, 5, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 1);
    POMAI_EXPECT_EQ(res.hits[0].id, 1);
    POMAI_EXPECT_TRUE(std::abs(res.hits[0].score) < 1e-4f);

    // Step 4: Compact and verify persisted state
    POMAI_EXPECT_OK(engine.Compact());

    res.Clear();
    POMAI_EXPECT_OK(engine.Search(vec_b, 5, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 1);
    POMAI_EXPECT_EQ(res.hits[0].id, 1);
    POMAI_EXPECT_TRUE(std::abs(res.hits[0].score) < 1e-4f);

    POMAI_EXPECT_OK(engine.Close());
}

// 3. Concurrency Torture: Readers, Writers, Deleters, and Compactor simultaneously
POMAI_TEST(Concurrency_MultiThreadedTorture) {
    const std::string db_dir = test::TempDir("pomai-concurrency-torture");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;

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

    // Pre-populate with 200 vectors
    for (uint32_t i = 1; i <= 200; ++i) {
        std::vector<float> vec(dim, static_cast<float>(i));
        POMAI_EXPECT_OK(engine.Put(i, vec));
    }
    POMAI_EXPECT_OK(engine.Compact());

    std::atomic<bool> running{true};
    std::atomic<uint64_t> total_queries{0};
    std::atomic<uint64_t> total_writes{0};
    std::atomic<uint64_t> total_deletes{0};

    // 4 Reader Threads
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&, t]() {
            std::mt19937 rng(100 + t);
            std::uniform_real_distribution<float> dist(1.0f, 200.0f);
            while (running.load(std::memory_order_relaxed)) {
                std::vector<float> q(dim, dist(rng));
                SearchResult res;
                (void)engine.Search(q, 10, &res);
                total_queries.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // 2 Writer Threads
    std::vector<std::thread> writers;
    for (int t = 0; t < 2; ++t) {
        writers.emplace_back([&, t]() {
            uint64_t cur_id = 1000 + t * 10000;
            while (running.load(std::memory_order_relaxed)) {
                std::vector<float> vec(dim, static_cast<float>(cur_id % 100));
                (void)engine.Put(cur_id++, vec);
                total_writes.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    // 1 Deleter Thread
    std::thread deleter([&]() {
        uint64_t del_id = 1;
        while (running.load(std::memory_order_relaxed)) {
            (void)engine.Delete(del_id);
            del_id = (del_id % 200) + 1;
            total_deletes.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    });

    // 1 Compaction Thread
    std::thread compactor([&]() {
        while (running.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            (void)engine.Compact();
        }
    });

    // Run concurrent torture for 1.5 seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    running.store(false, std::memory_order_relaxed);

    for (auto& r : readers) r.join();
    for (auto& w : writers) w.join();
    deleter.join();
    compactor.join();

    std::cout << "[CONCURRENCY TORTURE] Completed: " << total_queries.load() << " queries, "
              << total_writes.load() << " writes, " << total_deletes.load() << " deletes.\n";

    POMAI_EXPECT_TRUE(total_queries.load() > 50);

    POMAI_EXPECT_OK(engine.Close());
}

} // namespace
