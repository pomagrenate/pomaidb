#pragma once

// psync_spinlock.h
// Ultra-low latency SpinLock & Fair TicketLock
// Target: Linux / macOS / Windows across x86-64, ARM64, RISC-V

#include "psync_platform.h"
#include <thread>

namespace psync {

/**
 * SpinLock: Low-overhead spinlock with adaptive exponential backoff.
 * Ideal for ultra-short (<50ns) critical sections where sleeping via OS futex
 * introduces unnecessary context-switch latency.
 */
class SpinLock {
public:
    constexpr SpinLock() : state_(0) {}

    SpinLock(const SpinLock&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;

    void lock() noexcept {
        // Fast path: test and set
        u32 expected = 0;
        if (atomic_compare_exchange(
            &state_,
            &expected,
            1u,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        )) {
            return;
        }

        // Contended path: adaptive spin with exponential pause backoff
        usize backoff = 1;
        while (true) {
            while (atomic_load(&state_, MemoryOrder::relaxed) != 0) {
                for (usize i = 0; i < backoff; ++i) {
                    cpu_pause();
                }
                if (backoff < 64) {
                    backoff <<= 1;
                } else {
                    std::this_thread::yield();
                }
            }

            expected = 0;
            if (atomic_compare_exchange(
                &state_,
                &expected,
                1u,
                MemoryOrder::acquire,
                MemoryOrder::relaxed
            )) {
                return;
            }
        }
    }

    bool try_lock() noexcept {
        u32 expected = 0;
        return atomic_compare_exchange(
            &state_,
            &expected,
            1u,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        );
    }

    void unlock() noexcept {
        atomic_store(&state_, 0u, MemoryOrder::release);
    }

private:
    volatile u32 state_;
};

/**
 * TicketLock: Fair FIFO spinlock.
 * Eliminates starvation by dispensing tickets in arrival order.
 */
class TicketLock {
public:
    constexpr TicketLock() : now_serving_(0), next_ticket_(0) {}

    TicketLock(const TicketLock&) = delete;
    TicketLock& operator=(const TicketLock&) = delete;

    void lock() noexcept {
        u32 my_ticket = atomic_fetch_add(&next_ticket_, 1u, MemoryOrder::relaxed);
        usize spin_count = 0;
        while (atomic_load(&now_serving_, MemoryOrder::acquire) != my_ticket) {
            cpu_pause();
            if (++spin_count > 256) {
                std::this_thread::yield();
                spin_count = 0;
            }
        }
    }

    bool try_lock() noexcept {
        u32 serving = atomic_load(&now_serving_, MemoryOrder::acquire);
        u32 next = atomic_load(&next_ticket_, MemoryOrder::relaxed);
        if (serving != next) return false;

        u32 expected = serving;
        return atomic_compare_exchange(
            &next_ticket_,
            &expected,
            serving + 1,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        );
    }

    void unlock() noexcept {
        atomic_fetch_add(&now_serving_, 1u, MemoryOrder::release);
    }

private:
    volatile u32 now_serving_;
    volatile u32 next_ticket_;
};

} // namespace psync
