#pragma once

// psync_once.h
// Zero-libc, futex-backed one-time initialization primitive.
// Equivalent to std::call_once and std::once_flag.

#include "psync_platform.h"
#include <utility>

namespace psync {

// ============================================================================
// ONCE FLAG
// ============================================================================

struct OnceFlag {
    // 0 = uninitialized, 1 = initializing, 2 = initialized
    volatile u32 state{0};

    constexpr OnceFlag() = default;
    OnceFlag(const OnceFlag&) = delete;
    OnceFlag& operator=(const OnceFlag&) = delete;
};

// ============================================================================
// CALL ONCE
// ============================================================================

template <typename Callable, typename... Args>
void call_once(OnceFlag& flag, Callable&& f, Args&&... args) {
    // Fast path: already initialized with acquire barrier
    if (atomic_load(&flag.state, MemoryOrder::acquire) == 2) {
        return;
    }

    // Slow path: contended or first-time initialization
    while (true) {
        u32 expected = 0u;
        if (atomic_compare_exchange(
            &flag.state,
            &expected,
            1u,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        )) {
            // Winning thread: execute the callable
            try {
                f(std::forward<Args>(args)...);
                atomic_store(&flag.state, 2u, MemoryOrder::release);
                futex::wake_all(&flag.state);
                return;
            } catch (...) {
                // Exceptional case: reset to uninitialized so another thread can attempt
                atomic_store(&flag.state, 0u, MemoryOrder::release);
                futex::wake_all(&flag.state);
                throw;
            }
        } else if (expected == 2) {
            // Completed by another thread while we were trying
            return;
        } else {
            // Another thread is currently initializing: park on futex
            futex::wait(&flag.state, 1);
        }
    }
}

} // namespace psync
