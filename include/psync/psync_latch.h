#pragma once

// psync_latch.h
// Futex-backed non-allocating CountDownLatch (std::latch drop-in)
// Target: Linux / macOS / Windows across x86-64, ARM64, RISC-V

#include "psync_platform.h"
#include <chrono>

namespace psync {

/**
 * Latch is a downward counter of type u32 which can be used to synchronize threads.
 *
 * The value of the counter is initialized on creation. Threads may block on the latch
 * until the counter is decremented to zero. Unlike std::latch, psync::Latch is backed
 * directly by native OS futex primitives with zero dynamic memory allocation.
 */
class Latch {
public:
    explicit constexpr Latch(u32 count) : count_(count) {}

    Latch(const Latch&) = delete;
    Latch& operator=(const Latch&) = delete;

    /**
     * Decrements the counter in a thread-safe manner by n.
     * If the counter reaches zero, wakes all waiting threads.
     */
    void count_down(u32 n = 1) noexcept {
        PSYNC_ASSERT(n > 0);
        u32 old = atomic_fetch_sub(&count_, n, MemoryOrder::acq_rel);
        if (old <= n) {
            // Count reached 0 (or lower under bug/oversubscription)
            futex::wake_all(&count_);
        }
    }

    /**
     * Returns true if the internal counter has reached zero.
     */
    bool try_wait() const noexcept {
        return atomic_load(&count_, MemoryOrder::acquire) == 0;
    }

    /**
     * Waits until the counter reaches zero.
     */
    void wait() const noexcept {
        while (true) {
            u32 current = atomic_load(&count_, MemoryOrder::acquire);
            if (current == 0) return;
            futex::wait(const_cast<volatile u32*>(&count_), current);
        }
    }

    /**
     * Decrements the counter by n and waits until the counter reaches zero.
     */
    void arrive_and_wait(u32 n = 1) noexcept {
        count_down(n);
        wait();
    }

    /**
     * Waits until the counter reaches zero or the timeout expires.
     */
    template <typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& rel_time) const noexcept {
        if (try_wait()) return true;

        auto deadline = std::chrono::steady_clock::now() + rel_time;
        while (true) {
            u32 current = atomic_load(&count_, MemoryOrder::acquire);
            if (current == 0) return true;

            auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return try_wait();
            }
            u64 remaining_ns = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count()
            );
            futex::wait_for(const_cast<volatile u32*>(&count_), current, remaining_ns);
        }
    }

private:
    volatile u32 count_;
};

} // namespace psync
