#include "tests/common/test_main.h"

#include "options.h"
#include "utils/memtable_sizing.h"
#include "utils/system_memory.h"

namespace pomai {

POMAI_TEST(MemtableSizing_ManualOverride) {
    DBOptions opt;
    opt.memtable_flush_threshold_mb = 128u;  // Manual override
    opt.dim = 512;
    
    uint64_t threshold = utils::CalculateDynamicMemtableThreshold(opt, opt.dim);
    
    // Manual override should be respected
    POMAI_EXPECT_EQ(threshold, 128ull * 1024 * 1024);
}

POMAI_TEST(MemtableSizing_DynamicSizing) {
    DBOptions opt;
    opt.memtable_flush_threshold_mb = 0u;  // Enable dynamic sizing
    opt.memtable_budget_pct = 0.15f;  // 15% budget
    opt.memtable_min_threshold_mb = 16u;
    opt.memtable_max_threshold_mb = 512u;
    opt.memtable_min_vectors_per_segment = 8192;
    opt.dim = 512;
    
    // Simulate 1GB available RAM
    uint64_t available_ram = 1024ull * 1024 * 1024;
    uint64_t threshold = utils::CalculateDynamicMemtableThreshold(opt, opt.dim, available_ram);
    
    // 15% of 1GB = 153.6MB, should be clamped to [16MB, 512MB]
    uint64_t expected_min = 16ull * 1024 * 1024;
    uint64_t expected_max = 512ull * 1024 * 1024;
    POMAI_EXPECT_TRUE(threshold >= expected_min);
    POMAI_EXPECT_TRUE(threshold <= expected_max);
    
    // Dimension floor: 8192 * 512 * 4 = 16,777,216 bytes (~16MB)
    uint64_t dim_floor = 8192ull * 512 * 4;
    POMAI_EXPECT_TRUE(threshold >= dim_floor);
}

POMAI_TEST(MemtableSizing_DimensionFloorBounding) {
    DBOptions opt;
    opt.memtable_flush_threshold_mb = 0u;  // Enable dynamic sizing
    opt.memtable_budget_pct = 0.15f;
    opt.memtable_min_threshold_mb = 16u;
    opt.memtable_max_threshold_mb = 32u;  // Low max threshold
    opt.memtable_min_vectors_per_segment = 8192;
    opt.dim = 10000;  // High dimension
    
    // Simulate limited RAM (256MB)
    uint64_t available_ram = 256ull * 1024 * 1024;
    uint64_t threshold = utils::CalculateDynamicMemtableThreshold(opt, opt.dim, available_ram);
    
    // Dimension floor: 8192 * 10000 * 4 = 327,680,000 bytes (~312MB)
    // This exceeds max_threshold (32MB), so should be downscaled
    uint64_t max_threshold = 32ull * 1024 * 1024;
    POMAI_EXPECT_TRUE(threshold <= max_threshold);
}

POMAI_TEST(MemtableSizing_FallbackOnDetectionFailure) {
    DBOptions opt;
    opt.memtable_flush_threshold_mb = 0u;  // Enable dynamic sizing
    opt.memtable_budget_pct = 0.15f;
    opt.memtable_min_threshold_mb = 16u;
    opt.memtable_max_threshold_mb = 512u;
    opt.memtable_min_vectors_per_segment = 8192;
    opt.dim = 512;
    
    // No available RAM provided - should use fallback
    uint64_t threshold = utils::CalculateDynamicMemtableThreshold(opt, opt.dim, std::nullopt);
    
    // Should use fallback (64MB) or calculated from OS detection
    uint64_t expected_min = 16ull * 1024 * 1024;
    uint64_t expected_max = 512ull * 1024 * 1024;
    POMAI_EXPECT_TRUE(threshold >= expected_min);
    POMAI_EXPECT_TRUE(threshold <= expected_max);
}

POMAI_TEST(SystemMemoryInfo_Detection) {
    auto mem_info = utils::GetSystemMemoryInfo();
    
    // If detection succeeds, validate the results
    if (mem_info) {
        POMAI_EXPECT_TRUE(mem_info->total_bytes > 0);
        POMAI_EXPECT_TRUE(mem_info->available_bytes > 0);
        POMAI_EXPECT_TRUE(mem_info->available_bytes <= mem_info->total_bytes);
    } else {
        // Detection failure is acceptable on some platforms
        // Skip logging in test environment
    }
}

}  // namespace pomai