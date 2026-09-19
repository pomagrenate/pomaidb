// tests/test_main.cc
// Comprehensive test suite for ptask work-stealing task scheduler.

#include "ptask/ptask.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <atomic>
#include <numeric>
#include <string>

using namespace ptask;

static size_t tests_passed = 0;
static size_t tests_failed = 0;

#define RUN_TEST(fn) \
    do { \
        std::cout << "Running test: " << #fn << "... "; \
        try { \
            fn(); \
            std::cout << "PASSED\n"; \
            tests_passed++; \
        } catch (const std::exception& e) { \
            std::cout << "FAILED (" << e.what() << ")\n"; \
            tests_failed++; \
        } catch (...) { \
            std::cout << "FAILED (unknown exception)\n"; \
            tests_failed++; \
        } \
    } while (0)

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); \
    } while (0)

#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))

// ============================================================================
// TEST CASES
// ============================================================================

void test_types_and_hardware_clamping() {
    uint32_t hw = hardware_threads();
    ASSERT_TRUE(hw >= 1);

    // Requesting 0 defaults to hardware threads
    ASSERT_EQ(clamp_workers(0), hw);

    // Extreme over-subscription request (e.g. 50,000 threads) is clamped
    uint32_t clamped = clamp_workers(50000, 2.0f);
    ASSERT_TRUE(clamped <= hw * 2);
    ASSERT_TRUE(clamped >= 1);
}

void test_chase_lev_deque_single_threaded() {
    WorkStealingDeque<int> deque(16);
    constexpr int N = 1000;

    for (int i = 0; i < N; ++i) {
        deque.push(i);
    }
    ASSERT_EQ(deque.size(), N);

    // Pops in LIFO order
    for (int i = N - 1; i >= 0; --i) {
        auto val = deque.pop();
        ASSERT_TRUE(val.has_value());
        ASSERT_EQ(*val, i);
    }
    ASSERT_TRUE(deque.empty());
}

void test_chase_lev_deque_multi_stealers() {
    WorkStealingDeque<int> deque(64);
    constexpr int total_items = 20000;
    std::atomic<bool> producer_done{false};
    std::atomic<int64_t> stolen_sum{0};
    std::atomic<int64_t> popped_sum{0};

    // 3 stealer threads
    std::vector<std::thread> stealers;
    for (int s = 0; s < 3; ++s) {
        stealers.emplace_back([&]() {
            while (!producer_done.load(std::memory_order_relaxed) || !deque.empty()) {
                if (auto val = deque.steal()) {
                    stolen_sum.fetch_add(*val, std::memory_order_relaxed);
                } else {
                    cpu_pause();
                }
            }
        });
    }

    // Producer pushes and pops some
    for (int i = 1; i <= total_items; ++i) {
        deque.push(i);
        if (i % 5 == 0) {
            if (auto val = deque.pop()) {
                popped_sum.fetch_add(*val, std::memory_order_relaxed);
            }
        }
    }
    producer_done.store(true, std::memory_order_release);

    for (auto& st : stealers) {
        st.join();
    }

    // Drain remainder
    while (auto val = deque.pop()) {
        popped_sum.fetch_add(*val, std::memory_order_relaxed);
    }

    int64_t expected_sum = static_cast<int64_t>(total_items) * (total_items + 1) / 2;
    int64_t actual_sum = stolen_sum.load() + popped_sum.load();
    ASSERT_EQ(actual_sum, expected_sum);
}

void test_thread_pool_basic_and_futures() {
    ThreadPool pool(4);
    ASSERT_TRUE(pool.worker_count() >= 1);

    // Value future
    auto f1 = pool.submit([]() { return 42; });
    auto f2 = pool.submit([]() { return std::string("ptask"); });
    
    ASSERT_EQ(f1.get(), 42);
    ASSERT_EQ(f2.get(), "ptask");

    // Void future
    std::atomic<bool> executed{false};
    auto f3 = pool.submit([&]() { executed.store(true); });
    f3.get();
    ASSERT_TRUE(executed.load());
}

void test_thread_pool_extreme_oversubscription() {
    // Clamped thread pool (e.g. 4 workers) handling 100,000 tasks
    ThreadPool pool(4);
    constexpr int num_tasks = 100000;
    std::atomic<int64_t> counter{0};

    for (int i = 0; i < num_tasks; ++i) {
        pool.spawn([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.wait_idle();
    ASSERT_EQ(counter.load(), static_cast<int64_t>(num_tasks));
}

void test_thread_pool_nested_fork_join() {
    // Tests recursive / nested task submission with cooperative stealing
    ThreadPool pool(2);

    auto fib = [&](auto& self, int n) -> int {
        if (n <= 1) return n;
        if (n < 10) {
            return self(self, n - 1) + self(self, n - 2);
        }
        auto left = pool.submit([&self, n]() { return self(self, n - 1); });
        int right = self(self, n - 2);
        return left.get() + right;
    };

    auto res = pool.submit([&]() {
        return fib(fib, 15);
    });

    ASSERT_EQ(res.get(), 610);
}

void test_parallel_for() {
    ThreadPool pool(4);
    constexpr size_t N = 100000;
    std::vector<int> data(N, 0);

    parallel_for(pool, size_t{0}, N, [&](size_t i) {
        data[i] = static_cast<int>(i * 2);
    });

    for (size_t i = 0; i < N; ++i) {
        ASSERT_EQ(data[i], static_cast<int>(i * 2));
    }
}

void test_parallel_reduce() {
    ThreadPool pool(4);
    constexpr int64_t N = 100000;

    int64_t sum = parallel_reduce(
        pool,
        int64_t{1},
        N + 1,
        int64_t{0},
        [](int64_t a, int64_t b) { return a + b; },
        [](int64_t i) { return i; }
    );

    int64_t expected = N * (N + 1) / 2;
    ASSERT_EQ(sum, expected);
}

void test_adaptive_concurrency_limiter() {
    AdaptiveConcurrencyLimiter::Config cfg;
    cfg.initial_limit = 4;
    cfg.min_limit = 2;
    cfg.max_limit = 8;
    cfg.target_latency_us = 1000;
    cfg.sample_window = 10;
    AdaptiveConcurrencyLimiter limiter(cfg);

    ASSERT_EQ(limiter.current_limit(), 4u);
    ASSERT_TRUE(limiter.try_acquire());
    ASSERT_EQ(limiter.in_flight(), 1u);

    // Simulate fast completions (below target SLA): limit should increase
    for (int i = 0; i < 15; ++i) {
        limiter.release(200); // 200us < 1000us
        limiter.acquire();
    }
    ASSERT_TRUE(limiter.current_limit() > 4u);

    // Simulate slow completions (exceeding target SLA): limit should decrease
    for (int i = 0; i < 20; ++i) {
        limiter.release(5000); // 5000us > 1000us
        limiter.acquire();
    }
    limiter.release();
    ASSERT_TRUE(limiter.current_limit() <= 5u);
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== PTASK UNIT & CONCURRENCY TEST SUITE ===\n\n";

    RUN_TEST(test_types_and_hardware_clamping);
    RUN_TEST(test_chase_lev_deque_single_threaded);
    RUN_TEST(test_chase_lev_deque_multi_stealers);
    RUN_TEST(test_thread_pool_basic_and_futures);
    RUN_TEST(test_thread_pool_extreme_oversubscription);
    RUN_TEST(test_thread_pool_nested_fork_join);
    RUN_TEST(test_parallel_for);
    RUN_TEST(test_parallel_reduce);
    RUN_TEST(test_adaptive_concurrency_limiter);

    std::cout << "\n=== TEST SUMMARY ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    return tests_failed > 0 ? 1 : 0;
}
