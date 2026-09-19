#pragma once

// psync_baton.h
// 4-byte fast notification event / baton
// Target: Linux / macOS / Windows across x86-64, ARM64, RISC-V

#include "psync_platform.h"
#include <chrono>

namespace psync {

/**
 * Baton is a lightweight (4-byte), high-performance synchronization primitive
 * for one-shot or auto-reset thread handoffs and event notification.
 *
 * Far faster and lighter than a Mutex + ConditionVariable pair.
 */
class Baton {
public:
    enum State : u32 {
        kEmpty   = 0,
        kWaiting = 1,
        kReady   = 2
    };

    constexpr Baton() : state_(kEmpty) {}

    Baton(const Baton&) = delete;
    Baton& operator=(const Baton&) = delete;

    /**
     * Post the notification, waking any waiter.
     */
    void post() noexcept {
        u32 prev = atomic_exchange(&state_, static_cast<u32>(kReady), MemoryOrder::release);
        if (prev == kWaiting) {
            futex::wake_all(&state_);
        }
    }

    /**
     * Wait indefinitely until the baton is posted.
     */
    void wait() noexcept {
        // Fast path: already posted
        if (atomic_load(&state_, MemoryOrder::acquire) == kReady) {
            return;
        }

        // Adaptive spin before kernel wait
        for (usize i = 0; i < 32; ++i) {
            if (atomic_load(&state_, MemoryOrder::acquire) == kReady) {
                return;
            }
            cpu_pause();
        }

        // Transition from kEmpty to kWaiting and sleep on futex
        u32 expected = kEmpty;
        (void)atomic_compare_exchange(
            &state_,
            &expected,
            static_cast<u32>(kWaiting),
            MemoryOrder::relaxed,
            MemoryOrder::relaxed
        );

        while (atomic_load(&state_, MemoryOrder::acquire) != kReady) {
            futex::wait(&state_, kWaiting);
        }
    }

    /**
     * Try to wait without blocking. Returns true if baton was already posted.
     */
    bool try_wait() noexcept {
        return atomic_load(&state_, MemoryOrder::acquire) == kReady;
    }

    /**
     * Wait with relative timeout. Returns true if posted before timeout.
     */
    template <typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& rel_time) noexcept {
        if (try_wait()) return true;

        auto deadline = std::chrono::steady_clock::now() + rel_time;

        // Adaptive spin
        for (usize i = 0; i < 16; ++i) {
            if (try_wait()) return true;
            cpu_pause();
        }

        u32 expected = kEmpty;
        (void)atomic_compare_exchange(
            &state_,
            &expected,
            static_cast<u32>(kWaiting),
            MemoryOrder::relaxed,
            MemoryOrder::relaxed
        );

        while (atomic_load(&state_, MemoryOrder::acquire) != kReady) {
            auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return try_wait();
            }
            u64 remaining_ns = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count()
            );
            futex::wait_for(&state_, kWaiting, remaining_ns);
        }
        return true;
    }

    /**
     * Reset the baton back to kEmpty state.
     */
    void reset() noexcept {
        atomic_store(&state_, static_cast<u32>(kEmpty), MemoryOrder::relaxed);
    }

    /**
     * Returns true if the baton is currently in the ready state.
     */
    bool is_ready() const noexcept {
        return atomic_load(&state_, MemoryOrder::acquire) == kReady;
    }

private:
    volatile u32 state_;
};

} // namespace psync
