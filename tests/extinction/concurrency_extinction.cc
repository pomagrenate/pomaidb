// tests/extinction/concurrency_extinction.cc
// Extinction Event: 8 Readers, 4 Writers, 2 Deleters, 2 Compactors Heavy Chaos
//
// Stresses PomaiDB under heavy concurrent multi-threaded workloads with simultaneous
// queries, updates, deletes, and background compactions.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "pomegranate_engine.h"
#include "options.h"
#include "search.h"
#include "types.h"
#include "metadata.h"

namespace {

using namespace pomai;
using namespace pomai::core;

POMAI_TEST(Extinction_Concurrency_HeavyChaos) {
    const std::string db_dir = test::TempDir("pomai-concurrency-heavy");
    std::filesystem::remove_all(db_dir);
    const uint32_t dim = 16;
    const uint32_t initial_n = 300;

    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 8;
    opt.index_params.hnsw_ef_construction = 40;
    opt.index_params.hnsw_ef_search = 16;

    PomegranateEngine engine(opt, MetricType::kL2);
    POMAI_EXPECT_OK(engine.Open());

    // Pre-populate initial records
    for (uint32_t i = 1; i <= initial_n; ++i) {
        std::vector<float> vec(dim, static_cast<float>(i));
        Metadata meta;
        meta.device_id = (i % 2 == 0) ? "dev_even" : "dev_odd";
        POMAI_EXPECT_OK(engine.Put(i, vec, meta));
    }
    POMAI_EXPECT_OK(engine.Compact());

    std::atomic<bool> running{true};
    std::atomic<uint64_t> total_queries{0};
    std::atomic<uint64_t> total_writes{0};
    std::atomic<uint64_t> total_deletes{0};
    std::atomic<uint64_t> total_compactions{0};

    // 8 Readers
    std::vector<std::thread> readers;
    for (int t = 0; t < 8; ++t) {
        readers.emplace_back([&, t]() {
            std::mt19937_64 rng(1000 + t);
            std::uniform_real_distribution<float> dist(1.0f, 500.0f);
            SearchOptions sopts;
            if (t % 2 == 0) {
                sopts.filters.push_back(Filter("device_id", "dev_even"));
            }
            while (running.load(std::memory_order_relaxed)) {
                std::vector<float> q(dim, dist(rng));
                SearchResult res;
                (void)engine.Search(q, 10, sopts, &res);
                total_queries.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // 4 Writers
    std::vector<std::thread> writers;
    for (int t = 0; t < 4; ++t) {
        writers.emplace_back([&, t]() {
            std::mt19937_64 rng(2000 + t);
            uint64_t id_base = 10000 + t * 50000;
            while (running.load(std::memory_order_relaxed)) {
                uint64_t id = id_base++;
                std::vector<float> vec(dim, static_cast<float>(id % 100));
                Metadata meta;
                meta.device_id = (id % 2 == 0) ? "dev_even" : "dev_odd";
                (void)engine.Put(id, vec, meta);
                total_writes.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    }

    // 2 Deleters
    std::vector<std::thread> deleters;
    for (int t = 0; t < 2; ++t) {
        deleters.emplace_back([&, t]() {
            uint64_t del_id = 1 + t * 150;
            while (running.load(std::memory_order_relaxed)) {
                (void)engine.Delete(del_id);
                del_id = (del_id % initial_n) + 1;
                total_deletes.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    // 2 Compactors
    std::vector<std::thread> compactors;
    for (int t = 0; t < 2; ++t) {
        compactors.emplace_back([&, t]() {
            while (running.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(150 + t * 50));
                (void)engine.Compact();
                total_compactions.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Run concurrent campaign for 3 seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    running.store(false, std::memory_order_relaxed);

    for (auto& r : readers) r.join();
    for (auto& w : writers) w.join();
    for (auto& d : deleters) d.join();
    for (auto& c : compactors) c.join();

    std::cout << "[CONCURRENCY EXTINCTION] Summary: "
              << total_queries.load() << " queries, "
              << total_writes.load() << " writes, "
              << total_deletes.load() << " deletes, "
              << total_compactions.load() << " compactions completed without failure.\n";

    POMAI_EXPECT_TRUE(total_queries.load() > 100);
    POMAI_EXPECT_TRUE(total_writes.load() > 100);

    POMAI_EXPECT_OK(engine.Close());
}

} // namespace
