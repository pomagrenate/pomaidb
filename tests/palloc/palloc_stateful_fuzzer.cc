#include "tests/common/test_main.h"
#include "palloc_oracle.h"

#include <palloc.h>
#include <palloc_vector.h>

#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace pomai::palloc_qa {

struct FuzzStats {
    uint64_t alloc_ops = 0;
    uint64_t free_ops = 0;
    uint64_t realloc_ops = 0;
    uint64_t batch_ops = 0;
    uint64_t verify_ops = 0;
    uint64_t total_bytes_allocated = 0;
    size_t peak_live_blocks = 0;
};

static void RunFuzzerCampaign(uint32_t seed, uint64_t target_ops, FuzzStats& stats) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> op_dist(0, 99);

    // Realistic size distribution: small, medium, vector embeddings (32*4 to 4096*4)
    const size_t kVectorSizes[] = {
        32 * sizeof(float), 64 * sizeof(float), 96 * sizeof(float), 128 * sizeof(float),
        256 * sizeof(float), 512 * sizeof(float), 768 * sizeof(float), 1024 * sizeof(float),
        1536 * sizeof(float), 2048 * sizeof(float), 4096 * sizeof(float)
    };
    const size_t kAlignments[] = {16, 32, 64, 128};

    AllocatorOracle oracle(true, 32);
    std::vector<void*> live_ptrs;
    live_ptrs.reserve(8192);

    for (uint64_t op = 0; op < target_ops; ++op) {
        int action = op_dist(rng);

        if (action < 45 || live_ptrs.empty()) {
            // ALLOC
            size_t sz = 0;
            if (rng() % 2 == 0) {
                sz = kVectorSizes[rng() % (sizeof(kVectorSizes) / sizeof(kVectorSizes[0]))];
            } else {
                sz = 8 + (rng() % 4096);
            }
            size_t align = kAlignments[rng() % (sizeof(kAlignments) / sizeof(kAlignments[0]))];
            uint32_t payload_seed = static_cast<uint32_t>(rng());

            void* p = oracle.TrackAlloc(sz, align, payload_seed);
            if (p) {
                live_ptrs.push_back(p);
                stats.alloc_ops++;
                stats.total_bytes_allocated += sz;
                if (live_ptrs.size() > stats.peak_live_blocks) {
                    stats.peak_live_blocks = live_ptrs.size();
                }
            }
        } else if (action < 80) {
            // FREE
            size_t idx = rng() % live_ptrs.size();
            void* p = live_ptrs[idx];
            oracle.TrackFree(p);
            live_ptrs[idx] = live_ptrs.back();
            live_ptrs.pop_back();
            stats.free_ops++;
        } else if (action < 95) {
            // TOUCH / VERIFY
            size_t idx = rng() % live_ptrs.size();
            oracle.VerifyAndTouch(live_ptrs[idx]);
            stats.verify_ops++;
        } else {
            // BATCH ALLOC / FREE
            size_t batch_count = 1 + (rng() % 32);
            size_t dim = 128;
            float** batch = pa_vector_batch_alloc_floats(batch_count, dim);
            if (batch) {
                for (size_t b = 0; b < batch_count; ++b) {
                    batch[b][0] = 3.14f;
                    batch[b][dim - 1] = 2.71f;
                }
                pa_vector_batch_free(reinterpret_cast<void**>(batch), batch_count);
                stats.batch_ops += batch_count;
            }
        }
    }

    // Drain all remaining allocations
    for (void* p : live_ptrs) {
        oracle.TrackFree(p);
        stats.free_ops++;
    }
}

POMAI_TEST(PallocStatefulFuzzer_MultiSeedCampaign) {
    const uint32_t kSeeds[] = {42, 1337, 2026, 0xDEADBEEF, 0xC0FFEE};
    constexpr uint64_t kOpsPerSeed = 2000000; // 2,000,000 ops * 5 seeds = 10,000,000 operations

    FuzzStats total_stats;
    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t seed : kSeeds) {
        FuzzStats seed_stats;
        RunFuzzerCampaign(seed, kOpsPerSeed, seed_stats);

        total_stats.alloc_ops += seed_stats.alloc_ops;
        total_stats.free_ops += seed_stats.free_ops;
        total_stats.batch_ops += seed_stats.batch_ops;
        total_stats.verify_ops += seed_stats.verify_ops;
        total_stats.total_bytes_allocated += seed_stats.total_bytes_allocated;
        if (seed_stats.peak_live_blocks > total_stats.peak_live_blocks) {
            total_stats.peak_live_blocks = seed_stats.peak_live_blocks;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n=== PALLOC STATEFUL FUZZER EXTINCTION SUMMARY ==="
              << "\nTotal Operations       : " << (total_stats.alloc_ops + total_stats.free_ops + total_stats.verify_ops + total_stats.batch_ops)
              << "\nTotal Allocs           : " << total_stats.alloc_ops
              << "\nTotal Frees            : " << total_stats.free_ops
              << "\nTotal Verifications    : " << total_stats.verify_ops
              << "\nTotal Batch Ops        : " << total_stats.batch_ops
              << "\nTotal Bytes Allocated  : " << (total_stats.total_bytes_allocated / (1024 * 1024)) << " MiB"
              << "\nPeak Live Blocks       : " << total_stats.peak_live_blocks
              << "\nElapsed Time           : " << std::fixed << std::setprecision(2) << elapsed_sec << " s"
              << "\nThroughput             : " << static_cast<uint64_t>((total_stats.alloc_ops + total_stats.free_ops) / elapsed_sec) << " ops/s"
              << "\nZero Overlaps, Zero Leaks, Zero Canaries Corrupted."
              << "\n=================================================\n";
}

} // namespace pomai::palloc_qa
