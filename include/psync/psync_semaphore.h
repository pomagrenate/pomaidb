#pragma once

// psync_semaphore.h
// Futex-backed lightweight CountingSemaphore & BinarySemaphore
// Target: Linux / macOS / Windows across x86-64, ARM64, RISC-V

#include "psync_platform.h"
#include <chrono>

namespace psync {

/**
 * CountingSemaphore is a lightweight synchronization primitive that controls
 * access to a shared resource pool.
 *
 * Implemented with a single 32-bit atomic word backed directly by native OS futex.
 */
template <u32 LeastMaxValue = 0x7fffffffu>
class CountingSemaphore {
public:
    static constexpr u32 max() noexcept {
        return LeastMaxValue;
    }

    explicit constexpr CountingSemaphore(u32 desired) : count_(desired) {
        PSYNC_ASSERT(desired <= LeastMaxValue);
    }

    CountingSemaphore(const CountingSemaphore&) = delete;
    CountingSemaphore& operator=(const CountingSemaphore&) = delete;

    /**
     * Atomically increases the internal counter by update and unparks that many waiters.
     */
    void release(u32 update = 1) noexcept {
        PSYNC_ASSERT(update > 0);
        atomic_fetch_add(&count_, update, MemoryOrder::release);
        futex::wake(&count_, static_cast<i32>(update));
    }

    /**
     * Try to atomically acquire a token without blocking.
     */
    bool try_acquire() noexcept {
        u32 current = atomic_load(&count_, MemoryOrder::relaxed);
        while (current > 0) {
            u32 expected = current;
            if (atomic_compare_exchange(
                &count_,
                &expected,
                current - 1,
                MemoryOrder::acquire,
                MemoryOrder::relaxed
            )) {
                return true;
            }
            current = expected;
        }
        return false;
    }

    /**
     * Atomically acquires a token, blocking on futex until one is available.
     */
    void acquire() noexcept {
        // Fast path: immediate acquisition
        if (try_acquire()) return;

        // Adaptive spin before sleeping
        for (usize i = 0; i < 32; ++i) {
            if (try_acquire()) return;
            cpu_pause();
        }

        while (true) {
            if (try_acquire()) return;

            u32 current = atomic_load(&count_, MemoryOrder::relaxed);
            if (current == 0) {
                futex::wait(&count_, 0);
            }
        }
    }

    /**
     * Tries to acquire a token with relative timeout.
     */
    template <typename Rep, typename Period>
    bool try_acquire_for(const std::chrono::duration<Rep, Period>& rel_time) noexcept {
        if (try_acquire()) return true;

        auto deadline = std::chrono::steady_clock::now() + rel_time;

        // Adaptive spin
        for (usize i = 0; i < 16; ++i) {
            if (try_acquire()) return true;
            cpu_pause();
        }

        while (true) {
            if (try_acquire()) return true;

            auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return try_acquire();
            }

            u32 current = atomic_load(&count_, MemoryOrder::relaxed);
            if (current == 0) {
                u64 remaining_ns = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count()
                );
                futex::wait_for(&count_, 0, remaining_ns);
            }
        }
    }

private:
    volatile u32 count_;
};

/**
 * BinarySemaphore: Specialization of CountingSemaphore with max value 1.
 */
using BinarySemaphore = CountingSemaphore<1>;

} // namespace psync
