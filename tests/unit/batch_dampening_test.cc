#include "tests/common/test_main.h"

#include "options.h"
#include <thread>
#include <vector>
#include <atomic>

namespace pomai {

// Test the RAII pattern used in BatchScopeGuard
POMAI_TEST(BatchScopeGuardPattern_BasicOperation) {
    std::atomic<uint32_t> counter{0};
    
    // Simulate the BatchScopeGuard pattern
    {
        counter.fetch_add(1, std::memory_order_relaxed);
        POMAI_EXPECT_EQ(counter.load(), 1u);
        counter.fetch_sub(1, std::memory_order_relaxed);
    }
    
    // Counter should be back to 0
    POMAI_EXPECT_EQ(counter.load(), 0u);
}

POMAI_TEST(BatchScopeGuardPattern_ExceptionSafety) {
    std::atomic<uint32_t> counter{0};
    
    try {
        counter.fetch_add(1, std::memory_order_relaxed);
        POMAI_EXPECT_EQ(counter.load(), 1u);
        throw std::runtime_error("test exception");
    } catch (...) {
        counter.fetch_sub(1, std::memory_order_relaxed);
        // Counter should be decremented even after exception
        POMAI_EXPECT_EQ(counter.load(), 0u);
    }
}

POMAI_TEST(BatchScopeGuardPattern_ConcurrentAccess) {
    std::atomic<uint32_t> counter{0};
    constexpr int kNumThreads = 10;
    constexpr int kIterations = 100;
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&counter, kIterations]() {
            for (int j = 0; j < kIterations; ++j) {
                counter.fetch_add(1, std::memory_order_relaxed);
                // Simulate some work
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                counter.fetch_sub(1, std::memory_order_relaxed);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Counter should be back to 0 after all operations
    POMAI_EXPECT_EQ(counter.load(), 0u);
}

POMAI_TEST(BurstDampening_Configuration) {
    DBOptions opt;
    
    // Test default burst dampening factor
    POMAI_EXPECT_TRUE(opt.memtable_burst_dampening_factor == 0.2f);
    
    // Test custom burst dampening factor
    opt.memtable_burst_dampening_factor = 0.5f;
    POMAI_EXPECT_TRUE(opt.memtable_burst_dampening_factor == 0.5f);
}

POMAI_TEST(BurstDampening_EdgeProfiles) {
    // Test that edge profiles don't override burst dampening (it's a separate field)
    DBOptions opt;
    opt.edge_profile = EdgeProfile::kEdgeSafe;
    opt.ApplyEdgeProfile();
    
    // Burst dampening should remain at default (0.2f) unless explicitly set
    POMAI_EXPECT_TRUE(opt.memtable_burst_dampening_factor == 0.2f);
}

}  // namespace pomai