#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "palloc_oracle.h"

#include <palloc_vector.h>
#include <palloc/arena_pomai.h>
#include "src/utils/palloc_page_pool.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <filesystem>
#include <iostream>

namespace pomai::palloc_qa {

POMAI_TEST(PallocVectorModule_VecPool_LifecycleAndMemorySafety) {
    // Test pa_vec_pool with object_size = 64, initial_count = 10
    pa_vec_pool_t* pool = pa_vec_pool_create(64, 10);
    POMAI_EXPECT_TRUE(pool != nullptr);

    std::vector<void*> allocated;
    for (size_t i = 0; i < 20; ++i) {
        void* p = pa_vec_pool_alloc(pool);
        POMAI_EXPECT_TRUE(p != nullptr);
        // Write data into object
        std::memset(p, 0xA5, 64);
        allocated.push_back(p);
    }

    // Free half of them
    for (size_t i = 0; i < 10; ++i) {
        pa_vec_pool_free(pool, allocated[i]);
    }

    // Re-allocate
    for (size_t i = 0; i < 10; ++i) {
        void* p = pa_vec_pool_alloc(pool);
        POMAI_EXPECT_TRUE(p != nullptr);
        std::memset(p, 0x5A, 64);
    }

    // Destroy pool - must NOT crash or follow corrupted page pointers!
    pa_vec_pool_destroy(pool);
}

POMAI_TEST(PallocVectorModule_VecPool_SmallObjectSize) {
    // If object_size < sizeof(void*), e.g. 1 or 4 bytes
    pa_vec_pool_t* pool = pa_vec_pool_create(4, 16);
    if (pool != nullptr) {
        void* p1 = pa_vec_pool_alloc(pool);
        if (p1) {
            std::memset(p1, 0x77, 4);
            pa_vec_pool_free(pool, p1);
        }
        pa_vec_pool_destroy(pool);
    }
}

POMAI_TEST(PallocVectorModule_VecArena_LifecycleAndCap) {
    pa_vec_arena_t* arena = pa_vec_arena_create(4096, 64);
    POMAI_EXPECT_TRUE(arena != nullptr);

    // Set max size cap = 16 KiB
    pa_vec_arena_set_max_size(arena, 16384);

    std::vector<void*> ptrs;
    size_t total_alloc = 0;
    while (true) {
        void* p = pa_vec_arena_alloc(arena, 512);
        if (!p) break;
        POMAI_EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
        std::memset(p, 0x33, 512);
        ptrs.push_back(p);
        total_alloc += 512;
    }

    POMAI_EXPECT_TRUE(total_alloc <= 16384);
    POMAI_EXPECT_TRUE(total_alloc > 0);

    // Reset arena
    pa_vec_arena_reset(arena);

    // Can allocate again after reset
    void* p_after = pa_vec_arena_alloc(arena, 512);
    POMAI_EXPECT_TRUE(p_after != nullptr);
    POMAI_EXPECT_EQ(reinterpret_cast<uintptr_t>(p_after) % 64, 0u);

    pa_vec_arena_destroy(arena);
}

POMAI_TEST(PallocVectorModule_PomaiArena_ConcurrencyAndFuse) {
    constexpr size_t kCapacity = 1024 * 1024; // 1 MB
    pa_arena_t* arena = p_arena_create_for_vector_ex(kCapacity, true /* shared */);
    POMAI_EXPECT_TRUE(arena != nullptr);

    constexpr int kThreads = 8;
    constexpr size_t kVecDim = 128;
    std::atomic<size_t> successful_allocs{0};
    std::atomic<size_t> failed_allocs{0};

    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t]() {
            for (int i = 0; i < 2000; ++i) {
                float* v = static_cast<float*>(p_arena_alloc_vector(arena, kVecDim, sizeof(float)));
                if (v != nullptr) {
                    POMAI_EXPECT_EQ(reinterpret_cast<uintptr_t>(v) % PA_ARENA_VECTOR_ALIGN, 0u);
                    // Write thread marker
                    v[0] = static_cast<float>(t);
                    v[kVecDim - 1] = static_cast<float>(i);
                    successful_allocs.fetch_add(1, std::memory_order_relaxed);
                } else {
                    failed_allocs.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    POMAI_EXPECT_TRUE(successful_allocs.load() > 0);
    // Hard limit must have triggered at some point
    POMAI_EXPECT_TRUE(failed_allocs.load() > 0);

    // Reset arena
    p_arena_reset(arena);

    // After reset, allocation succeeds again
    float* v_after = static_cast<float*>(p_arena_alloc_vector(arena, kVecDim, sizeof(float)));
    POMAI_EXPECT_TRUE(v_after != nullptr);
    POMAI_EXPECT_EQ(reinterpret_cast<uintptr_t>(v_after) % PA_ARENA_VECTOR_ALIGN, 0u);

    p_arena_destroy(arena);
}

POMAI_TEST(PallocVectorModule_BatchAlloc_Extremes) {
    // 1. count = 0
    void** b0 = pa_vector_batch_alloc(0, 128);
    POMAI_EXPECT_TRUE(b0 == nullptr);

    float** f0 = pa_vector_batch_alloc_floats(0, 128);
    POMAI_EXPECT_TRUE(f0 == nullptr);

    // 2. Huge count overflow
    void** b_huge = pa_vector_batch_alloc(std::numeric_limits<size_t>::max() / sizeof(void*) + 1, 64);
    POMAI_EXPECT_TRUE(b_huge == nullptr);

    // 3. Normal batch alloc
    constexpr size_t kCount = 64;
    constexpr size_t kDim = 256;
    float** vecs = pa_vector_batch_alloc_floats(kCount, kDim);
    POMAI_EXPECT_TRUE(vecs != nullptr);
    for (size_t i = 0; i < kCount; ++i) {
        POMAI_EXPECT_TRUE(vecs[i] != nullptr);
        POMAI_EXPECT_EQ(reinterpret_cast<uintptr_t>(vecs[i]) % 64, 0u);
        vecs[i][0] = 1.0f;
        vecs[i][kDim - 1] = 2.0f;
    }
    pa_vector_batch_free(reinterpret_cast<void**>(vecs), kCount);
}

POMAI_TEST(PallocVectorModule_PagePool_ConcurrentStress) {
    palloc_page_pool_options opts{};
    opts.page_size = 4096;
    opts.capacity_bytes = 4096 * 16; // 16 pages
    opts.swap_file_path = nullptr;
    opts.device_profile = PALLOC_DEVICE_PROFILE_SMALL;

    palloc_page_pool* pool = palloc_page_pool_create(&opts);
    POMAI_EXPECT_TRUE(pool != nullptr);

    constexpr int kThreads = 4;
    std::atomic<bool> stop{false};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int op = 0; op < 500; ++op) {
                uint64_t page_id = (t * 10) + (op % 10);
                int is_new = 0;
                void* p = palloc_fetch_page(pool, page_id, 1, &is_new);
                if (p) {
                    std::memset(p, 0x11 * (t + 1), 4096);
                    palloc_unpin_page(pool, page_id, 1);
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    palloc_page_pool_destroy(pool);
}

} // namespace pomai::palloc_qa
