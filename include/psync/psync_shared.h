#pragma once

// psync_shared.h
// 4-byte shared mutex (reader/writer lock)
// Target: Linux x86-64 / Windows
// Compilers: GCC, Clang, MSVC

#include "psync_platform.h"
#include "psync_locks.h"

namespace psync {

// ============================================================================
// SHARED MUTEX STATE LAYOUT (32 bits)
// ============================================================================
// SHARED MUTEX STATE LAYOUT (32 bits)
// ============================================================================
// Bit 0:       WRITER_LOCKED flag (1 = writer holds lock, 0 = no writer)
// Bits 1-15:   WRITER_WAITING (number of waiting writers, up to 32767)
// Bits 16-31:  READER_COUNT (number of active readers, up to 65535)
// ============================================================================

class SharedMutex {
public:
    // State constants
    static constexpr u32 UNLOCKED = 0x00000000u;
    static constexpr u32 WRITER_LOCKED = 0x00000001u;
    static constexpr u32 WRITER_FLAG = 0x00000001u;
    static constexpr u32 WRITER_WAITING_SHIFT = 1;
    static constexpr u32 WRITER_WAITING_INC = 0x00000002u;
    static constexpr u32 WRITER_WAITING_MASK = 0x0000FFFEu;
    
    static constexpr u32 READER_SHIFT = 16;
    static constexpr u32 READER_INCREMENT = 0x00010000u;
    static constexpr u32 READER_MASK = 0xFFFF0000u;
    static constexpr u32 MAX_READERS = 0x0000FFFFu;  // 65535 readers
    
    // Initialize shared mutex (trivial initialization)
    constexpr SharedMutex() : state_(UNLOCKED) {}
    
    // Disable copy and move
    SharedMutex(const SharedMutex&) = delete;
    SharedMutex& operator=(const SharedMutex&) = delete;
    
    // Acquire the mutex for exclusive (writer) access
    void lock() {
        // Fast path: try to acquire uncontended lock
        u32 expected = UNLOCKED;
        if (atomic_compare_exchange(
            &state_,
            &expected,
            WRITER_LOCKED,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        )) {
            // Successfully acquired lock uncontended
            return;
        }
        
        // Slow path: contended acquisition
        lock_contended();
    }
    
    // Try to acquire the mutex for exclusive (writer) access without blocking
    // Returns true if lock was acquired, false otherwise
    bool try_lock() {
        u32 expected = UNLOCKED;
        return atomic_compare_exchange(
            &state_,
            &expected,
            WRITER_LOCKED,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        );
    }
    
    // Release the mutex from exclusive (writer) access
    void unlock() {
        // Clear writer locked flag
        atomic_fetch_and(&state_, ~WRITER_LOCKED, MemoryOrder::release);
        
        // Wake all waiters (readers and writers)
        futex::wake_all(&state_);
    }
    
    // Acquire the mutex for shared (reader) access
    void lock_shared() {
        // Fast path: try to acquire uncontended read lock
        u32 current = atomic_load(&state_, MemoryOrder::relaxed);
        
        while (true) {
            // If writer holds lock OR writers are waiting, go to slow path to prevent writer starvation
            if ((current & (WRITER_LOCKED | WRITER_WAITING_MASK)) != 0) {
                lock_shared_contended();
                return;
            }
            
            // Try to increment reader count
            u32 new_state = current + READER_INCREMENT;
            u32 expected = current;
            
            if (atomic_compare_exchange(
                &state_,
                &expected,
                new_state,
                MemoryOrder::acquire,
                MemoryOrder::relaxed
            )) {
                // Successfully acquired read lock
                return;
            }
            
            // CAS failed, retry
            current = expected;
        }
    }
    
    // Try to acquire the mutex for shared (reader) access without blocking
    // Returns true if lock was acquired, false otherwise
    bool try_lock_shared() {
        u32 current = atomic_load(&state_, MemoryOrder::relaxed);
        while (true) {
            // Check if writer is locked or waiting
            if ((current & (WRITER_LOCKED | WRITER_WAITING_MASK)) != 0) {
                return false;
            }
            
            // Try to increment reader count
            u32 new_state = current + READER_INCREMENT;
            u32 expected = current;
            
            if (atomic_compare_exchange(
                &state_,
                &expected,
                new_state,
                MemoryOrder::acquire,
                MemoryOrder::relaxed
            )) {
                return true;
            }
            current = expected;
        }
    }
    
    // Release the mutex from shared (reader) access
    void unlock_shared() {
        // Decrement reader count
        u32 prev = atomic_fetch_sub(&state_, READER_INCREMENT, MemoryOrder::release);
        
        // If this was the last reader and writers are waiting, wake them
        u32 reader_count = ((prev & READER_MASK) >> READER_SHIFT) - 1;
        if (reader_count == 0 && (prev & WRITER_WAITING_MASK) != 0) {
            futex::wake_all(&state_);
        }
    }
    
private:
    // Contended exclusive (writer) lock acquisition
    void lock_contended() {
        // Register this writer as waiting
        atomic_fetch_add(&state_, WRITER_WAITING_INC, MemoryOrder::relaxed);
        
        usize spin_count = 0;
        usize backoff = SPIN_INITIAL;
        
        while (true) {
            u32 current = atomic_load(&state_, MemoryOrder::relaxed);
            
            // Lock can be acquired if no writer holds it and no active readers
            if ((current & (WRITER_LOCKED | READER_MASK)) == 0) {
                u32 new_state = (current - WRITER_WAITING_INC) | WRITER_LOCKED;
                u32 expected = current;
                if (atomic_compare_exchange(
                    &state_,
                    &expected,
                    new_state,
                    MemoryOrder::acquire,
                    MemoryOrder::relaxed
                )) {
                    return;
                }
                continue;
            }
            
            // Adaptive spin before sleeping
            if (spin_count < FUTEX_THRESHOLD) {
                for (usize i = 0; i < backoff; ++i) {
                    cpu_pause();
                }
                spin_count++;
                backoff = (backoff * BACKOFF_MULTIPLIER > BACKOFF_MAX) ? BACKOFF_MAX : backoff * BACKOFF_MULTIPLIER;
                continue;
            }
            
            // Wait on futex
            futex::wait(&state_, current);
            
            // Reset spin count after waking up
            spin_count = 0;
            backoff = SPIN_INITIAL;
        }
    }
    
    // Contended shared (reader) lock acquisition
    void lock_shared_contended() {
        usize spin_count = 0;
        usize backoff = SPIN_INITIAL;
        
        while (true) {
            u32 current = atomic_load(&state_, MemoryOrder::relaxed);
            
            // Can acquire if no writer is locked AND no writers are waiting
            if ((current & (WRITER_LOCKED | WRITER_WAITING_MASK)) == 0) {
                u32 new_state = current + READER_INCREMENT;
                u32 expected = current;
                if (atomic_compare_exchange(
                    &state_,
                    &expected,
                    new_state,
                    MemoryOrder::acquire,
                    MemoryOrder::relaxed
                )) {
                    return;
                }
                continue;
            }
            
            // Adaptive spin before sleeping
            if (spin_count < FUTEX_THRESHOLD) {
                for (usize i = 0; i < backoff; ++i) {
                    cpu_pause();
                }
                spin_count++;
                backoff = (backoff * BACKOFF_MULTIPLIER > BACKOFF_MAX) ? BACKOFF_MAX : backoff * BACKOFF_MULTIPLIER;
                continue;
            }
            
            // Wait on futex
            futex::wait(&state_, current);
            
            // Reset spin count after waking up
            spin_count = 0;
            backoff = SPIN_INITIAL;
        }
    }
    
    volatile u32 state_;
};

// ============================================================================
// SHARED LOCK GUARD (RAII for readers)
// ============================================================================

class SharedLockGuard {
public:
    explicit SharedLockGuard(SharedMutex& mutex) : mutex_(mutex) {
        mutex_.lock_shared();
    }
    
    ~SharedLockGuard() {
        mutex_.unlock_shared();
    }
    
    // Disable copy and move
    SharedLockGuard(const SharedLockGuard&) = delete;
    SharedLockGuard& operator=(const SharedLockGuard&) = delete;
    
private:
    SharedMutex& mutex_;
};

// ============================================================================
// EXCLUSIVE LOCK GUARD (RAII for writers)
// ============================================================================

class UniqueLockGuard {
public:
    explicit UniqueLockGuard(SharedMutex& mutex) : mutex_(mutex) {
        mutex_.lock();
    }
    
    ~UniqueLockGuard() {
        mutex_.unlock();
    }
    
    // Disable copy and move
    UniqueLockGuard(const UniqueLockGuard&) = delete;
    UniqueLockGuard& operator=(const UniqueLockGuard&) = delete;
    
private:
    SharedMutex& mutex_;
};

} // namespace psync
