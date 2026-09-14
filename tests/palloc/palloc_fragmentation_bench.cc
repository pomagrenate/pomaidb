#include "tests/common/test_main.h"

#include <palloc.h>
#include <palloc_vector.h>

#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>
#include <iomanip>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <psapi.h>
#endif

namespace pomai::palloc_qa {

static size_t GetProcessRssBytes() {
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#endif
    return 0;
}

POMAI_TEST(PallocFragmentation_AdversarialHolesAndReuse) {
    constexpr size_t kLargeSize = 64 * 1024;
    constexpr size_t kSmallSize = 256;
    constexpr size_t kNumPairs = 2000;

    size_t rss_initial = GetProcessRssBytes();

    std::vector<void*> large_ptrs;
    std::vector<void*> small_ptrs;
    large_ptrs.reserve(kNumPairs);
    small_ptrs.reserve(kNumPairs);

    size_t total_requested = 0;
    size_t total_usable = 0;

    for (size_t i = 0; i < kNumPairs; ++i) {
        void* l = pa_malloc(kLargeSize);
        void* s = pa_malloc(kSmallSize);
        POMAI_EXPECT_TRUE(l != nullptr && s != nullptr);

        std::memset(l, 0x11, 128);
        std::memset(s, 0x22, 128);

        total_requested += kLargeSize + kSmallSize;
        total_usable += pa_usable_size(l) + pa_usable_size(s);

        large_ptrs.push_back(l);
        small_ptrs.push_back(s);
    }

    double internal_frag = static_cast<double>(total_usable - total_requested) / static_cast<double>(total_requested);
    size_t rss_peak = GetProcessRssBytes();

    // Free all small blocks creating thousands of 256-byte holes
    for (void* s : small_ptrs) {
        pa_free(s);
    }
    small_ptrs.clear();

    // Allocate 10,000 new small blocks (256 bytes) - palloc MUST reuse the freed holes!
    for (size_t i = 0; i < 10000; ++i) {
        void* p = pa_malloc(kSmallSize);
        POMAI_EXPECT_TRUE(p != nullptr);
        std::memset(p, 0x33, 128);
        small_ptrs.push_back(p);
    }

    size_t rss_after_reuse = GetProcessRssBytes();

    // Free everything
    for (void* l : large_ptrs) pa_free(l);
    for (void* s : small_ptrs) pa_free(s);

    // Trigger full collect
    pa_collect(true);
    size_t rss_final = GetProcessRssBytes();

    std::cout << "\n=== PALLOC FRAGMENTATION & RSS AUDIT ==="
              << "\nInternal Fragmentation : " << std::fixed << std::setprecision(2) << (internal_frag * 100.0) << " %"
              << "\nInitial RSS            : " << (rss_initial / (1024 * 1024)) << " MiB"
              << "\nPeak RSS               : " << (rss_peak / (1024 * 1024)) << " MiB"
              << "\nRSS After Hole Reuse   : " << (rss_after_reuse / (1024 * 1024)) << " MiB"
              << "\nFinal RSS After Collect: " << (rss_final / (1024 * 1024)) << " MiB"
              << "\n========================================\n";

    POMAI_EXPECT_TRUE(internal_frag < 0.25);
}

POMAI_TEST(PallocFragmentation_VectorWorkloadChurn) {
    const size_t kDims[] = {32, 64, 128, 256, 512, 1024, 1536};
    constexpr int kCycles = 50;
    constexpr int kAllocsPerCycle = 1000;

    std::vector<float*> live_vectors;
    live_vectors.reserve(kAllocsPerCycle * 2);

    for (int cycle = 0; cycle < kCycles; ++cycle) {
        for (int i = 0; i < kAllocsPerCycle; ++i) {
            size_t dim = kDims[i % (sizeof(kDims) / sizeof(kDims[0]))];
            float* v = pa_vector_alloc_floats(dim);
            POMAI_EXPECT_TRUE(v != nullptr);
            POMAI_EXPECT_EQ(reinterpret_cast<uintptr_t>(v) % 64, 0u);
            v[0] = static_cast<float>(cycle);
            live_vectors.push_back(v);
        }

        // Keep 20% live, free 80%
        size_t to_free = (live_vectors.size() * 4) / 5;
        for (size_t j = 0; j < to_free; ++j) {
            pa_free(live_vectors[j]);
        }
        live_vectors.erase(live_vectors.begin(), live_vectors.begin() + to_free);
    }

    for (float* v : live_vectors) {
        pa_free(v);
    }
    pa_collect(true);
}

} // namespace pomai::palloc_qa
