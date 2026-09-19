#pragma once

// psync_mutex.h
// 4-byte adaptive mutex
// Target: Linux x86-64 / Windows
// Compilers: GCC, Clang, MSVC

#include "psync_platform.h"
#include "psync_locks.h"

namespace psync {

// Forward declaration for ConditionVariable
class ConditionVariable;

// ============================================================================
// MUTEX STATE LAYOUT (32 bits)
// ============================================================================
// Bits 31-16: Reserved (future use)
// Bits 15-1:  WAITER_COUNT (number of parked waiters)
// Bit 0:      LOCKED flag (1 = locked, 0 = unlocked)
// ============================================================================

class Mutex {
public:
    // State constants
    static constexpr u32 UNLOCKED = 0x00000000u;
    static constexpr u32 LOCKED_NO_WAITERS = 0x00000001u;
    static constexpr u32 LOCKED_FLAG = 0x00000001u;
    static constexpr u32 WAITER_SHIFT = 1;
    static constexpr u32 WAITER_INCREMENT = 0x00000002u;  // 1 << WAITER_SHIFT
    
    // Initialize mutex (trivial initialization)
    constexpr Mutex() : state_(UNLOCKED) {}
    
    // Disable copy and move
    Mutex(const Mutex&) = delete;
    Mutex& operator=(const Mutex&) = delete;
    
    // Acquire the mutex
    // Fast path: userspace CAS
    // Slow path: adaptive spin + futex wait
    void lock() {
        // Fast path: try to acquire uncontended lock
        u32 expected = UNLOCKED;
        if (atomic_compare_exchange(
            &state_,
            &expected,
            LOCKED_NO_WAITERS,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        )) {
            // Successfully acquired lock uncontended
            return;
        }
        
        // Slow path: contended acquisition
        lock_contended();
    }
    
    // Try to acquire the mutex without blocking
    // Returns true if lock was acquired, false otherwise
    bool try_lock() {
        u32 expected = UNLOCKED;
        return atomic_compare_exchange(
            &state_,
            &expected,
            LOCKED_NO_WAITERS,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        );
    }
    
    // Release the mutex
    // Fast path: CAS if no waiters
    // Slow path: clear locked flag and wake one waiter
    void unlock() {
        u32 current = atomic_load(&state_, MemoryOrder::relaxed);
        
        // Fast path: no waiters present
        if (current == LOCKED_NO_WAITERS) {
            u32 expected = LOCKED_NO_WAITERS;
            if (atomic_compare_exchange(
                &state_,
                &expected,
                UNLOCKED,
                MemoryOrder::release,
                MemoryOrder::relaxed
            )) {
                // Successfully unlocked with no waiters
                return;
            }
            // CAS failed: state changed, retry with slow path
            current = atomic_load(&state_, MemoryOrder::relaxed);
        }
        
        // Slow path: waiters present
        unlock_contended(current);
    }
    
private:
    // Get state address for futex requeue (used by ConditionVariable)
    // This provides direct access to the mutex state for FUTEX_REQUEUE
    volatile u32* get_state_address() {
        return &state_;
    }
    
    volatile u32 state_;  // Mutex state
    
    // Contended lock acquisition with adaptive spinning
    void lock_contended() {
        usize spin_count = 0;
        usize backoff = SPIN_INITIAL;
        
        while (true) {
            // Adaptive spin loop
            for (usize i = 0; i < backoff; ++i) {
                cpu_pause();
                
                // Try to acquire lock
                u32 expected = UNLOCKED;
                if (atomic_compare_exchange(
                    &state_,
                    &expected,
                    LOCKED_NO_WAITERS,
                    MemoryOrder::acquire,
                    MemoryOrder::relaxed
                )) {
                    // Successfully acquired lock after spinning
                    return;
                }
            }
            
            // Increase backoff
            spin_count++;
            if (spin_count < FUTEX_THRESHOLD) {
                backoff = backoff * BACKOFF_MULTIPLIER;
                if (backoff > BACKOFF_MAX) {
                    backoff = BACKOFF_MAX;
                }
                continue;
            }
            
            // After threshold, register as waiter and sleep
            atomic_fetch_add(&state_, WAITER_INCREMENT, MemoryOrder::relaxed);
            
            while (true) {
                u32 current = atomic_load(&state_, MemoryOrder::relaxed);
                
                // If lock is unlocked, acquire it AND decrement our waiter count
                if ((current & LOCKED_FLAG) == 0) {
                    u32 new_state = (current - WAITER_INCREMENT) | LOCKED_FLAG;
                    u32 expected = current;
                    if (atomic_compare_exchange(
                        &state_,
                        &expected,
                        new_state,
                        MemoryOrder::acquire,
                        MemoryOrder::relaxed
                    )) {
                        // Successfully acquired lock and removed waiter
                        return;
                    }
                    // CAS failed, retry
                    continue;
                }
                
                // Lock is held, wait on futex
                futex::wait(&state_, current);
            }
        }
    }
    
    // Contended unlock with waiter wake
    void unlock_contended(u32 current) {
        (void)current;
        // Clear locked flag atomically
        atomic_fetch_and(&state_, ~LOCKED_FLAG, MemoryOrder::release);
        
        // Wake one waiter if there are any
        futex::wake(&state_, 1);
    }
    
    // Allow ConditionVariable to access get_state_address
    friend class ConditionVariable;
};

} // namespace psync