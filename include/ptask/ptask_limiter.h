#pragma once

// ptask_limiter.h
// Adaptive Concurrency Limiter & Admission Control (AIMD / Gradient feedback).
// Prevents system saturation and latency spikes under extreme load.
// Part of ptask - High-performance work-stealing task scheduler.

#include "ptask_types.h"
#include <atomic>
#include <chrono>
#include <algorithm>

namespace ptask {

struct LimiterConfig {
    uint32_t min_limit{2};
    uint32_t max_limit{64};
    uint32_t initial_limit{8};
    uint64_t target_latency_us{5000};  // 5ms target SLA
    float backoff_factor{0.8f};        // Multiplicative decrease factor
    uint32_t sample_window{100};       // Samples before adjusting limit
};

class AdaptiveConcurrencyLimiter {
public:
    using Config = LimiterConfig;

    AdaptiveConcurrencyLimiter() : AdaptiveConcurrencyLimiter(Config{}) {}

    explicit AdaptiveConcurrencyLimiter(Config config)
        : config_(config),
          current_limit_(config.initial_limit),
          in_flight_(0),
          sample_count_(0),
          sample_latency_sum_(0) {
        if (config_.initial_limit == 0) {
            config_.initial_limit = hardware_threads();
        }
        if (config_.max_limit == 0) {
            config_.max_limit = hardware_threads() * 2;
        }
        current_limit_.store(config_.initial_limit, std::memory_order_relaxed);
    }

    // Try to acquire an execution slot without blocking.
    // Returns true if admitted, false if backpressure rejected.
    bool try_acquire() noexcept {
        uint32_t limit = current_limit_.load(std::memory_order_relaxed);
        uint32_t curr = in_flight_.load(std::memory_order_relaxed);
        while (curr < limit) {
            if (in_flight_.compare_exchange_weak(curr, curr + 1,
                                                std::memory_order_acquire,
                                                std::memory_order_relaxed)) {
                return true;
            }
        }
        return false;
    }

    // Blocking / yielding acquire with backoff
    void acquire() noexcept {
        while (!try_acquire()) {
            cpu_pause();
        }
    }

    // Release slot and record latency sample to adjust limit
    void release(uint64_t latency_us) noexcept {
        in_flight_.fetch_sub(1, std::memory_order_release);

        sample_latency_sum_.fetch_add(latency_us, std::memory_order_relaxed);
        uint32_t count = sample_count_.fetch_add(1, std::memory_order_relaxed) + 1;

        if (count >= config_.sample_window) {
            // Winning thread adjusts limit
            if (sample_count_.compare_exchange_strong(count, 0, std::memory_order_relaxed)) {
                uint64_t total_latency = sample_latency_sum_.exchange(0, std::memory_order_relaxed);
                uint64_t avg_latency = total_latency / count;
                adjust_limit(avg_latency);
            }
        }
    }

    // Release slot without recording latency
    void release() noexcept {
        in_flight_.fetch_sub(1, std::memory_order_release);
    }

    uint32_t current_limit() const noexcept {
        return current_limit_.load(std::memory_order_relaxed);
    }

    uint32_t in_flight() const noexcept {
        return in_flight_.load(std::memory_order_relaxed);
    }

private:
    void adjust_limit(uint64_t avg_latency_us) noexcept {
        uint32_t current = current_limit_.load(std::memory_order_relaxed);
        uint32_t new_limit = current;

        if (avg_latency_us <= config_.target_latency_us) {
            // Additive increase
            new_limit = std::min(current + 1, config_.max_limit);
        } else {
            // Multiplicative decrease
            uint32_t decreased = static_cast<uint32_t>(static_cast<float>(current) * config_.backoff_factor);
            new_limit = std::max(decreased, config_.min_limit);
        }

        current_limit_.store(new_limit, std::memory_order_relaxed);
    }

    Config config_;
    std::atomic<uint32_t> current_limit_;
    std::atomic<uint32_t> in_flight_;
    std::atomic<uint32_t> sample_count_;
    std::atomic<uint64_t> sample_latency_sum_;
};

} // namespace ptask
