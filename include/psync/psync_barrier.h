#pragma once

// psync_barrier.h
// Cyclic, phased synchronization barrier (std::barrier equivalent)
// Target: Linux / macOS / Windows across x86-64, ARM64, RISC-V

#include "psync_platform.h"
#include <functional>

namespace psync {

/**
 * Barrier is a coordination mechanism that enables multiple threads to synchronize
 * across repeated phases.
 *
 * Each phase blocks participating threads until the expected number of threads
 * have arrived. When the last thread arrives, an optional completion function is executed,
 * the phase counter is incremented, and all waiting threads are unparked.
 */
class Barrier {
public:
    explicit Barrier(u32 expected_count)
        : expected_(expected_count),
          arrived_(expected_count),
          phase_(0)
    {
        PSYNC_ASSERT(expected_count > 0);
    }

    Barrier(const Barrier&) = delete;
    Barrier& operator=(const Barrier&) = delete;

    /**
     * Arrive at the barrier and wait for all other expected threads in this phase.
     */
    void arrive_and_wait() noexcept {
        u32 current_phase = atomic_load(&phase_, MemoryOrder::acquire);
        u32 old = atomic_fetch_sub(&arrived_, 1u, MemoryOrder::acq_rel);

        if (old == 1) {
            // Last thread to arrive: advance phase and reset arrival counter
            atomic_store(&arrived_, expected_, MemoryOrder::relaxed);
            atomic_fetch_add(&phase_, 1u, MemoryOrder::release);
            futex::wake_all(&phase_);
        } else {
            // Wait for phase to change
            while (atomic_load(&phase_, MemoryOrder::acquire) == current_phase) {
                futex::wait(&phase_, current_phase);
            }
        }
    }

    /**
     * Arrive without waiting. Decrements the arrival counter and returns the arrival token.
     */
    u32 arrive() noexcept {
        u32 current_phase = atomic_load(&phase_, MemoryOrder::acquire);
        u32 old = atomic_fetch_sub(&arrived_, 1u, MemoryOrder::acq_rel);
        if (old == 1) {
            atomic_store(&arrived_, expected_, MemoryOrder::relaxed);
            atomic_fetch_add(&phase_, 1u, MemoryOrder::release);
            futex::wake_all(&phase_);
        }
        return current_phase;
    }

    /**
     * Wait for the completion of the phase associated with arrival_phase.
     */
    void wait(u32 arrival_phase) noexcept {
        while (atomic_load(&phase_, MemoryOrder::acquire) == arrival_phase) {
            futex::wait(&phase_, arrival_phase);
        }
    }

private:
    const u32 expected_;
    volatile u32 arrived_;
    volatile u32 phase_;
};

} // namespace psync
