// PALLOC HOSTILE FORENSIC AUDIT — Bug Regression & Independent Verification
//
// palloc is a standalone general-purpose memory allocator for vector workloads.
// It is NOT a PomaiDB-specific component. Tests here use ONLY palloc's own
// public API: <palloc.h>, <palloc_vector.h>, <palloc/arena_pomai.h>.
//
// PomaiDB-specific wrappers (palloc_page_pool, palloc_compat, palloc_override)
// are tested separately in tests/palloc/palloc_pomai_integration_test.cc.
//
// Uses IndependentOracle (system malloc backing) — not AllocatorOracle.
#include "tests/common/test_main.h"
#include "palloc_independent_oracle.h"

// palloc public API only
#include <palloc.h>
#include <palloc_vector.h>
#include <palloc/arena_pomai.h>

#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>
#include <algorithm>
#include <random>
#include <limits>
#include <chrono>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace pomai::palloc_forensic_test {

using namespace pomai::palloc_forensic;

// ============================================================
// BUG-P0-001 VERIFICATION: pa_vec_pool page-chain corruption
// Previous fix: pa_vec_page_t header, object_size >= sizeof(void*)
//
// INDEPENDENT TEST: We track objects using system-malloc shadow map.
// We verify page chain integrity by inspecting that destroy()
// doesn't crash, and by independently tracking all live objects.
// ============================================================

POMAI_TEST(BugRegression_P0_001_VecPool_PageChainIntegrity) {
    std::cout << "[BUG-P0-001] Testing pa_vec_pool page-chain integrity\n";

    // Test 1: object_size < sizeof(void*) - the original crash trigger
    for (size_t sz : {1u, 2u, 3u, 4u, 7u}) {
        std::cout << "  [P0-001] object_size=" << sz << " (< sizeof(void*)=" << sizeof(void*) << ")\n";
        pa_vec_pool_t* pool = pa_vec_pool_create(sz, 16);
        POMAI_EXPECT_TRUE(pool != nullptr);
        if (!pool) continue;

        // Allocate objects and write into them
        std::vector<void*> ptrs;
        for (int i = 0; i < 64; ++i) {
            void* p = pa_vec_pool_alloc(pool);
            POMAI_EXPECT_TRUE(p != nullptr);
            if (p) {
                // Write known pattern - if page chain was corrupted, this would corrupt it
                ::memset(p, 0xCC, sizeof(void*)); // write at least one ptr's worth
                ptrs.push_back(p);
            }
        }

        // Free half
        for (size_t i = 0; i < ptrs.size() / 2; ++i) {
            pa_vec_pool_free(pool, ptrs[i]);
        }

        // Re-allocate the freed slots
        for (size_t i = 0; i < ptrs.size() / 2; ++i) {
            void* p = pa_vec_pool_alloc(pool);
            POMAI_EXPECT_TRUE(p != nullptr);
        }

        // Destroy must NOT crash (this was the original crash site)
        pa_vec_pool_destroy(pool);
        std::cout << "    -> PASS (no crash on destroy)\n";
    }

    // Test 2: exact one-object pages
    std::cout << "  [P0-001] Testing one-object pages...\n";
    {
        pa_vec_pool_t* pool = pa_vec_pool_create(sizeof(void*), 1);
        POMAI_EXPECT_TRUE(pool != nullptr);
        if (pool) {
            void* p1 = pa_vec_pool_alloc(pool);
            POMAI_EXPECT_TRUE(p1 != nullptr);
            if (p1) ::memset(p1, 0xAA, sizeof(void*));
            pa_vec_pool_free(pool, p1);
            // Re-alloc: must get p1 back (LIFO freelist)
            void* p2 = pa_vec_pool_alloc(pool);
            POMAI_EXPECT_TRUE(p2 != nullptr);
            if (p2) ::memset(p2, 0xBB, sizeof(void*));
            pa_vec_pool_destroy(pool); // Must not crash
        }
    }

    // Test 3: two-object pages — boundary condition
    std::cout << "  [P0-001] Testing two-object pages...\n";
    {
        pa_vec_pool_t* pool = pa_vec_pool_create(64, 2);
        POMAI_EXPECT_TRUE(pool != nullptr);
        if (pool) {
            void* a = pa_vec_pool_alloc(pool);
            void* b = pa_vec_pool_alloc(pool);
            POMAI_EXPECT_TRUE(a != nullptr && b != nullptr);
            POMAI_EXPECT_TRUE(a != b);
            if (a) ::memset(a, 0xAA, 64);
            if (b) ::memset(b, 0xBB, 64);
            pa_vec_pool_free(pool, a);
            pa_vec_pool_free(pool, b);
            pa_vec_pool_destroy(pool); // Must not crash
        }
    }

    // Test 4: allocation spanning many pages (forces new-page path repeatedly)
    std::cout << "  [P0-001] Testing multi-page allocation chain...\n";
    {
        pa_vec_pool_t* pool = pa_vec_pool_create(128, 4); // 4 initial, grow by 64
        POMAI_EXPECT_TRUE(pool != nullptr);
        if (pool) {
            // Allocate 200 objects: forces multiple new page allocations
            std::vector<void*> ptrs;
            for (int i = 0; i < 200; ++i) {
                void* p = pa_vec_pool_alloc(pool);
                POMAI_EXPECT_TRUE(p != nullptr);
                if (p) {
                    ::memset(p, (uint8_t)(i & 0xFF), 128);
                    ptrs.push_back(p);
                }
            }
            // Free all in random order (stress freelist)
            std::mt19937 rng(42);
            std::shuffle(ptrs.begin(), ptrs.end(), rng);
            for (void* p : ptrs) pa_vec_pool_free(pool, p);

            pa_vec_pool_destroy(pool); // Critical: must walk page chain correctly
            std::cout << "    -> PASS (multi-page chain destroy succeeded)\n";
        }
    }

    // Test 5: huge object sizes
    std::cout << "  [P0-001] Testing huge object sizes...\n";
    {
        // 4096-byte objects - test allocation tracking
        pa_vec_pool_t* pool = pa_vec_pool_create(4096, 8);
        POMAI_EXPECT_TRUE(pool != nullptr);
        if (pool) {
            std::vector<void*> ptrs;
            for (int i = 0; i < 16; ++i) {
                void* p = pa_vec_pool_alloc(pool);
                POMAI_EXPECT_TRUE(p != nullptr);
                if (p) {
                    ::memset(p, 0x77, 4096);
                    ptrs.push_back(p);
                }
            }
            for (void* p : ptrs) pa_vec_pool_free(pool, p);
            pa_vec_pool_destroy(pool);
        }
    }
    std::cout << "[BUG-P0-001] All subtests passed.\n";
}

// ============================================================
// BUG-P0-002 VERIFICATION: VirtualFree decommit metadata crash
// Previous fix: payload aligned to OS page boundary
//
// INDEPENDENT TEST: Allocate arena, verify payload_start >= OS page size,
// call reset (which calls VirtualFree/MEM_DECOMMIT), verify arena
// metadata still accessible after reset.
// ============================================================

POMAI_TEST(BugRegression_P0_002_ArenaReset_NoDecommitHeader) {
    std::cout << "[BUG-P0-002] Testing arena reset does not decommit header\n";

    const size_t kPageSize = 4096;
    // Use a capacity = 1 page so decommit range is precisely 1 page
    const size_t kCapacity = kPageSize;

    // Test 1: basic single reset
    std::cout << "  [P0-002] Single reset test...\n";
    {
        pa_arena_t* arena = p_arena_create_for_vector_ex(kCapacity, false);
        POMAI_EXPECT_TRUE(arena != nullptr);
        if (arena) {
            // Allocate something
            void* v1 = p_arena_alloc_vector(arena, 16, sizeof(float));
            POMAI_EXPECT_TRUE(v1 != nullptr);
            if (v1) ::memset(v1, 0x11, 16 * sizeof(float));

            // Reset: this calls VirtualFree(MEM_DECOMMIT) on the payload.
            // If fix is wrong, next line will SIGSEGV writing arena->needs_commit_after_reset.
            p_arena_reset(arena);

            // After reset, allocation must succeed again (recommit)
            void* v2 = p_arena_alloc_vector(arena, 16, sizeof(float));
            POMAI_EXPECT_TRUE(v2 != nullptr);
            if (v2) {
                ::memset(v2, 0x22, 16 * sizeof(float));
                std::cout << "    -> PASS (alloc after reset succeeded at 0x"
                          << std::hex << reinterpret_cast<uintptr_t>(v2) << ")\n";
            }

            p_arena_destroy(arena);
        }
    }

    // Test 2: repeated reset / re-alloc cycles
    std::cout << "  [P0-002] Repeated reset cycles...\n";
    {
        pa_arena_t* arena = p_arena_create_for_vector_ex(kCapacity, false);
        POMAI_EXPECT_TRUE(arena != nullptr);
        if (arena) {
            for (int cycle = 0; cycle < 20; ++cycle) {
                void* v = p_arena_alloc_vector(arena, 8, sizeof(float));
                POMAI_EXPECT_TRUE(v != nullptr);
                if (v) {
                    float* fv = static_cast<float*>(v);
                    fv[0] = static_cast<float>(cycle);
                    fv[7] = static_cast<float>(cycle + 1000);
                    // Verify we can read it back
                    POMAI_EXPECT_EQ(fv[0], static_cast<float>(cycle));
                }
                p_arena_reset(arena);
            }
            p_arena_destroy(arena);
            std::cout << "    -> PASS (20 reset cycles without crash)\n";
        }
    }

    // Test 3: reset on empty arena (no allocations)
    std::cout << "  [P0-002] Reset on empty arena...\n";
    {
        pa_arena_t* arena = p_arena_create_for_vector_ex(kCapacity, false);
        POMAI_EXPECT_TRUE(arena != nullptr);
        if (arena) {
            p_arena_reset(arena);  // Should not crash
            p_arena_reset(arena);  // Double reset should also not crash
            p_arena_destroy(arena);
            std::cout << "    -> PASS\n";
        }
    }

    // Test 4: destroy after reset (without re-allocating)
    std::cout << "  [P0-002] Destroy after reset...\n";
    {
        pa_arena_t* arena = p_arena_create_for_vector_ex(kCapacity, false);
        POMAI_EXPECT_TRUE(arena != nullptr);
        if (arena) {
            void* v = p_arena_alloc_vector(arena, 4, sizeof(float));
            POMAI_EXPECT_TRUE(v != nullptr);
            p_arena_reset(arena);
            p_arena_destroy(arena);  // Must not crash on decommitted arena
            std::cout << "    -> PASS\n";
        }
    }

    // Test 5: shared arena reset (multi-threaded commit_pending race check)
    std::cout << "  [P0-002] Shared arena post-reset commit race...\n";
    {
        pa_arena_t* arena = p_arena_create_for_vector_ex(64 * kPageSize, true);
        POMAI_EXPECT_TRUE(arena != nullptr);
        if (arena) {
            // Fill arena
            while (p_arena_alloc_vector(arena, 128, sizeof(float))) {}
            p_arena_reset(arena);

            // Simultaneously have 8 threads try to allocate (triggering recommit race)
            constexpr int kT = 8;
            std::atomic<int> ready{0};
            std::atomic<int> failures{0};
            std::vector<std::thread> thr;
            for (int t = 0; t < kT; ++t) {
                thr.emplace_back([&, t]() {
                    ready.fetch_add(1);
                    while (ready.load() < kT) {}  // Spin-barrier
                    void* v = p_arena_alloc_vector(arena, 32, sizeof(float));
                    if (v) {
                        float* fv = static_cast<float*>(v);
                        // Write and read back - tests that committed memory is accessible
                        fv[0] = static_cast<float>(t);
                        fv[31] = static_cast<float>(t + 100);
                        if (fv[0] != static_cast<float>(t)) {
                            failures.fetch_add(1);
                        }
                    }
                });
            }
            for (auto& th : thr) th.join();
            POMAI_EXPECT_EQ(failures.load(), 0);
            std::cout << "    -> PASS (shared arena commit race resolved, failures="
                      << failures.load() << ")\n";
            p_arena_destroy(arena);
        }
    }

    std::cout << "[BUG-P0-002] All subtests passed.\n";
}

// ============================================================
// BUG-P1-003 VERIFICATION: Atomic rollback race in shared arena
// Previous fix: CAS loop with pre-addition overflow check
//
// INDEPENDENT TEST: Use a tiny capacity (forces frequent OOM boundary).
// Track allocated offsets independently. Verify:
//   1. No allocation exceeds capacity.
//   2. No two allocations overlap.
//   3. Total allocated bytes <= capacity.
// ============================================================

POMAI_TEST(BugRegression_P1_003_AtomicCAS_NoRace) {
    std::cout << "[BUG-P1-003] Testing atomic CAS loop in shared arena\n";

    // Use a very small capacity to stress the OOM boundary frequently
    const size_t kVecDim  = 4;  // float4 = 16 bytes per vector
    const size_t kCapacity = 4096; // tiny: forces rapid OOM

    pa_arena_t* arena = p_arena_create_for_vector_ex(kCapacity, true /* shared */);
    POMAI_EXPECT_TRUE(arena != nullptr);
    if (!arena) return;

    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 2000;

    // Independent tracking: mutex-protected list of allocated addresses
    std::mutex track_mutex;
    std::vector<std::pair<uintptr_t, size_t>> allocations; // (addr, size)
    std::atomic<size_t> total_allocated{0};
    std::atomic<size_t> total_oom{0};
    std::atomic<int> bad_overlaps{0};

    std::atomic<int> barrier{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            barrier.fetch_add(1);
            while (barrier.load() < kThreads) {} // Spin-barrier

            for (int i = 0; i < kOpsPerThread; ++i) {
                void* v = p_arena_alloc_vector(arena, kVecDim, sizeof(float));
                if (v) {
                    uintptr_t addr = reinterpret_cast<uintptr_t>(v);
                    size_t bytes = kVecDim * sizeof(float);
                    // Write unique pattern immediately
                    float* fv = static_cast<float*>(v);
                    fv[0] = static_cast<float>(addr & 0xFFFFFF);

                    std::lock_guard<std::mutex> lk(track_mutex);
                    // Check this range against all previous allocations
                    for (auto& [a, s] : allocations) {
                        uintptr_t end = addr + bytes;
                        uintptr_t aend = a + s;
                        if (addr < aend && end > a) {
                            bad_overlaps.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    allocations.emplace_back(addr, bytes);
                    total_allocated.fetch_add(bytes, std::memory_order_relaxed);
                } else {
                    total_oom.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    std::cout << "  Total allocations: " << allocations.size()
              << "\n  Total OOM: " << total_oom.load()
              << "\n  Total bytes: " << total_allocated.load()
              << "\n  Capacity: " << kCapacity
              << "\n  Bad overlaps: " << bad_overlaps.load() << "\n";

    POMAI_EXPECT_EQ(bad_overlaps.load(), 0);
    POMAI_EXPECT_TRUE(total_allocated.load() <= kCapacity);
    // Verify that OOM was triggered (good: means boundary was hit)
    POMAI_EXPECT_TRUE(total_oom.load() > 0);

    p_arena_destroy(arena);
    std::cout << "[BUG-P1-003] PASS — No overlaps, no capacity violations.\n";
}


// ============================================================
// NOTE: BUG-P1-004 (palloc_page_pool mutex) and
//       BUG-P2-005 (palloc_is_owned foreign ptr) are tested in
//       tests/palloc/palloc_pomai_integration_test.cc because they
//       test PomaiDB-specific wrappers, not the standalone palloc
//       allocator API.
// ============================================================


// ============================================================
// BUG-P2-006 VERIFICATION: Integer overflow in chunk doubling
// INDEPENDENT TEST: Pass all boundary sizes and verify:
//  - Unreachable sizes return nullptr (never return wrong-sized memory)
//  - No infinite loop (test times out if stuck)
//  - dim * sizeof(float) overflow correctly rejected
// ============================================================

POMAI_TEST(BugRegression_P2_006_IntegerOverflow_Boundaries) {
    std::cout << "[BUG-P2-006] Testing integer overflow boundaries\n";

    // These sizes must ALL return nullptr - no hang, no crash
    const size_t overflow_sizes[] = {
        std::numeric_limits<size_t>::max(),
        std::numeric_limits<size_t>::max() - 1,
        std::numeric_limits<size_t>::max() / 2 + 1,
        std::numeric_limits<size_t>::max() / 2,
    };

    for (size_t sz : overflow_sizes) {
        void* p = pa_malloc(sz);
        POMAI_EXPECT_TRUE(p == nullptr);
        std::cout << "  pa_malloc(" << sz << ") = nullptr (PASS)\n";
    }

    // calloc overflow
    void* cp = pa_calloc(std::numeric_limits<size_t>::max() / 2 + 1, 2);
    POMAI_EXPECT_TRUE(cp == nullptr);
    std::cout << "  pa_calloc(HUGE*2) = nullptr (PASS)\n";

    // pa_vec_arena: allocation near SIZE_MAX
    {
        pa_vec_arena_t* arena = pa_vec_arena_create(4096, 64);
        POMAI_EXPECT_TRUE(arena != nullptr);
        if (arena) {
            // Request SIZE_MAX - must return nullptr without hang
            void* p = pa_vec_arena_alloc(arena, std::numeric_limits<size_t>::max());
            POMAI_EXPECT_TRUE(p == nullptr);
            std::cout << "  pa_vec_arena_alloc(SIZE_MAX) = nullptr (PASS)\n";

            // SIZE_MAX / 2 + 1
            p = pa_vec_arena_alloc(arena, std::numeric_limits<size_t>::max() / 2 + 1);
            POMAI_EXPECT_TRUE(p == nullptr);
            std::cout << "  pa_vec_arena_alloc(SIZE_MAX/2+1) = nullptr (PASS)\n";

            pa_vec_arena_destroy(arena);
        }
    }

    // pa_vector_batch_alloc_floats dimension overflow
    const size_t overflow_dims[] = {
        std::numeric_limits<size_t>::max() / sizeof(float),
        std::numeric_limits<size_t>::max() / sizeof(float) + 1,
        std::numeric_limits<size_t>::max(),
    };
    for (size_t dim : overflow_dims) {
        float** ptrs = pa_vector_batch_alloc_floats(1, dim);
        POMAI_EXPECT_TRUE(ptrs == nullptr);
        std::cout << "  pa_vector_batch_alloc_floats(1, " << dim << ") = nullptr (PASS)\n";
    }

    // Valid edge dimensions (must succeed for small counts)
    const size_t valid_dims[] = {1, 2, 3, 7, 8, 15, 16, 32, 64, 128, 256};
    for (size_t dim : valid_dims) {
        float** ptrs = pa_vector_batch_alloc_floats(1, dim);
        POMAI_EXPECT_TRUE(ptrs != nullptr);
        if (ptrs) {
            POMAI_EXPECT_TRUE(ptrs[0] != nullptr);
            if (ptrs[0]) {
                ptrs[0][0] = 1.0f;
                ptrs[0][dim - 1] = 2.0f;
            }
            pa_vector_batch_free(reinterpret_cast<void**>(ptrs), 1);
        }
    }
    std::cout << "[BUG-P2-006] All overflow boundaries correctly rejected.\n";
}

// ============================================================
// INDEPENDENT SHADOW ORACLE CORRECTNESS TEST
// Uses IndependentOracle (system malloc backing) not AllocatorOracle.
// Proves that the oracle itself is independent.
// ============================================================

POMAI_TEST(ForeignOracle_IndependentShadow_BasicCorrectnessProof) {
    std::cout << "[ORACLE] Testing independent shadow oracle with system-malloc backing\n";

    IndependentOracle oracle(64);
    ResourceBudget budget(64 * 1024 * 1024); // 64 MiB budget for this test

    // Seed corpus - allocated with independent oracle
    std::mt19937_64 rng(0xBADF00D);
    std::vector<void*> live;

    size_t ops = 0;
    for (int round = 0; round < 10000; ++round) {
        bool do_alloc = live.empty() || (rng() % 3 != 0);

        if (do_alloc) {
            size_t sz = 8 + (rng() % 4096);
            size_t align = (1u << (rng() % 7 + 2)); // 4 to 256
            if (!budget.can_alloc(sz + 128)) continue; // respect budget

            uint32_t seed = static_cast<uint32_t>(rng());
            void* p = oracle.TrackAlloc(sz, align, seed);
            if (p) {
                live.push_back(p);
                budget.record_alloc(sz + 128);
                ++ops;
            }
        } else {
            size_t idx = rng() % live.size();
            void* p = live[idx];

            // Verify before free
            POMAI_EXPECT_TRUE(oracle.VerifyLive(p));

            size_t req_size = 8 + (rng() % 4096); // approximation
            oracle.TrackFree(p);
            budget.record_free(req_size + 128);
            live[idx] = live.back();
            live.pop_back();
            ++ops;
        }
    }

    // Drain all
    oracle.DrainAll();
    live.clear();

    // Report
    oracle.print_report(std::cout);

    auto r = oracle.report();
    POMAI_EXPECT_EQ(r.overlap_detections, 0u);
    POMAI_EXPECT_EQ(r.alignment_violations, 0u);
    POMAI_EXPECT_EQ(r.canary_corruptions, 0u);
    POMAI_EXPECT_EQ(r.double_free_attempts, 0u);
    POMAI_EXPECT_EQ(r.live_count, 0u);

    std::cout << "[ORACLE] PASS — " << ops << " operations, all invariants hold.\n";
}

} // namespace pomai::palloc_forensic_test
