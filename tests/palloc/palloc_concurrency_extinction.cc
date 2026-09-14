#include "tests/common/test_main.h"
#include "palloc_oracle.h"

#include <palloc.h>
#include <palloc_vector.h>
#include <palloc/arena_pomai.h>

#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace pomai::palloc_qa {

POMAI_TEST(PallocConcurrency_ThreadMigration_CrossThreadFree) {
    // Thread A allocates -> Thread B frees
    // Thread B allocates -> Thread C frees
    // Thread C allocates -> Thread A frees
    constexpr int kNumThreads = 4;
    constexpr int kOperationsPerThread = 10000;

    struct WorkItem {
        void* ptr;
        size_t size;
        uint32_t marker;
    };

    std::vector<std::queue<WorkItem>> queues(kNumThreads);
    std::vector<std::mutex> mutexes(kNumThreads);
    std::vector<std::condition_variable> cvs(kNumThreads);
    std::atomic<bool> done{false};

    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            int next_t = (t + 1) % kNumThreads;
            std::mt19937 rng(42 + t);
            std::uniform_int_distribution<size_t> size_dist(16, 4096);

            for (int i = 0; i < kOperationsPerThread; ++i) {
                // 1. Allocate block
                size_t sz = size_dist(rng);
                void* p = pa_malloc_aligned(sz, 16);
                POMAI_EXPECT_TRUE(p != nullptr);
                uint32_t marker = static_cast<uint32_t>((t << 24) | (i & 0xFFFFFF));
                std::memset(p, static_cast<uint8_t>(marker & 0xFF), sz);

                // 2. Enqueue to next thread
                {
                    std::lock_guard<std::mutex> lock(mutexes[next_t]);
                    queues[next_t].push({p, sz, marker});
                }
                cvs[next_t].notify_one();

                // 3. Dequeue and free from our queue
                WorkItem item{nullptr, 0, 0};
                {
                    std::unique_lock<std::mutex> lock(mutexes[t]);
                    if (cvs[t].wait_for(lock, std::chrono::milliseconds(5), [&]() { return !queues[t].empty(); })) {
                        item = queues[t].front();
                        queues[t].pop();
                    }
                }
                if (item.ptr) {
                    // Verify marker
                    uint8_t exp = static_cast<uint8_t>(item.marker & 0xFF);
                    POMAI_EXPECT_EQ(static_cast<uint8_t*>(item.ptr)[0], exp);
                    POMAI_EXPECT_EQ(static_cast<uint8_t*>(item.ptr)[item.size - 1], exp);
                    pa_free(item.ptr);
                }
            }

            // Drain remaining
            while (true) {
                WorkItem item{nullptr, 0, 0};
                {
                    std::unique_lock<std::mutex> lock(mutexes[t]);
                    if (queues[t].empty()) break;
                    item = queues[t].front();
                    queues[t].pop();
                }
                if (item.ptr) {
                    uint8_t exp = static_cast<uint8_t>(item.marker & 0xFF);
                    POMAI_EXPECT_EQ(static_cast<uint8_t*>(item.ptr)[0], exp);
                    pa_free(item.ptr);
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // Drain any remaining across all queues
    for (int t = 0; t < kNumThreads; ++t) {
        while (!queues[t].empty()) {
            WorkItem item = queues[t].front();
            queues[t].pop();
            pa_free(item.ptr);
        }
    }
}

POMAI_TEST(PallocConcurrency_HighContentionStress) {
    // 16 threads rapidly allocating, reallocating, and freeing simultaneously
    constexpr int kThreads = 16;
    constexpr int kOps = 20000;
    std::atomic<bool> start_flag{false};

    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t]() {
            while (!start_flag.load(std::memory_order_acquire)) {}

            std::mt19937 rng(1337 + t);
            std::uniform_int_distribution<size_t> size_dist(8, 8192);
            std::uniform_int_distribution<int> op_dist(0, 99);

            std::vector<void*> live;
            live.reserve(512);

            for (int i = 0; i < kOps; ++i) {
                int op = op_dist(rng);
                if (op < 50 || live.empty()) {
                    // Alloc
                    size_t sz = size_dist(rng);
                    void* p = pa_malloc(sz);
                    POMAI_EXPECT_TRUE(p != nullptr);
                    static_cast<uint8_t*>(p)[0] = static_cast<uint8_t>(t);
                    static_cast<uint8_t*>(p)[sz - 1] = static_cast<uint8_t>(t);
                    live.push_back(p);
                } else if (op < 80) {
                    // Free
                    size_t idx = rng() % live.size();
                    pa_free(live[idx]);
                    live[idx] = live.back();
                    live.pop_back();
                } else {
                    // Realloc
                    size_t idx = rng() % live.size();
                    size_t new_sz = size_dist(rng);
                    void* p = pa_realloc(live[idx], new_sz);
                    POMAI_EXPECT_TRUE(p != nullptr);
                    static_cast<uint8_t*>(p)[new_sz - 1] = static_cast<uint8_t>(t);
                    live[idx] = p;
                }
            }

            for (void* p : live) {
                pa_free(p);
            }
        });
    }

    start_flag.store(true, std::memory_order_release);
    for (auto& w : workers) {
        w.join();
    }
}

} // namespace pomai::palloc_qa
