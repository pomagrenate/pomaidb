#pragma once

// psync_tagged_ptr.h
// Zero-byte overhead lock using tagged pointer with alignment bits
// Target: Linux x86-64 / Windows
// Compilers: GCC, Clang, MSVC

#include "psync_platform.h"
#include "psync_microlock.h"

namespace psync {

// ============================================================================
// TAGGED LOCK POINTER
// ============================================================================
// Uses low-order alignment bits for lock state and waiter flags
// Assumes 8-byte alignment (3 low bits are naturally zero)
// Bit 0: Locked state
// Bit 1: Waiters present flag
// Bit 2: Reserved for user/Marked flag
// Bits 3-63: Actual pointer value (NO SHIFTING - direct bitmasking)
// ============================================================================

template<typename T>
class TaggedLockPtr {
public:
    // Bit layout (direct bitmasking, no shifting)
    static constexpr u64 LOCKED_BIT = 0x1ull;       // Bit 0
    static constexpr u64 WAITERS_BIT = 0x2ull;     // Bit 1
    static constexpr u64 MARKED_BIT = 0x4ull;      // Bit 2 (reserved for user)
    static constexpr u64 FLAG_MASK = 0x7ull;       // Bits 0-2 (all flags)
    static constexpr u64 POINTER_MASK = ~FLAG_MASK; // Bits 3-63 (pointer)
    
    // Initialize with null pointer
    constexpr TaggedLockPtr() : tagged_ptr_(0) {}
    
    // Initialize with pointer (requires 8-byte alignment)
    explicit TaggedLockPtr(T* ptr) {
        PSYNC_ASSERT((reinterpret_cast<u64>(ptr) & FLAG_MASK) == 0);
        tagged_ptr_ = reinterpret_cast<u64>(ptr);
    }
    
    // Disable copy and move
    TaggedLockPtr(const TaggedLockPtr&) = delete;
    TaggedLockPtr& operator=(const TaggedLockPtr&) = delete;
    
    // Get raw pointer (atomically, with direct bitmasking)
    T* get() const {
        u64 current = atomic_load(&tagged_ptr_, MemoryOrder::acquire);
        return reinterpret_cast<T*>(current & POINTER_MASK);
    }
    
    // Lock acquisition (uncontended fast path)
    void lock() {
        while (true) {
            u64 current = atomic_load(&tagged_ptr_, MemoryOrder::relaxed);
            
            // Check if already locked
            if ((current & LOCKED_BIT) != 0) {
                // Contended path
                lock_contended();
                return;
            }
            
            // Try to set locked bit
            u64 new_state = current | LOCKED_BIT;
            u64 expected = current;
            if (atomic_compare_exchange(
                &tagged_ptr_,
                &expected,
                new_state,
                MemoryOrder::acquire,
                MemoryOrder::relaxed
            )) {
                // Successfully acquired lock
                return;
            }
            // CAS failed, retry
        }
    }
    
    // Try lock without blocking
    bool try_lock() {
        u64 current = atomic_load(&tagged_ptr_, MemoryOrder::relaxed);
        
        // Check if already locked
        if ((current & LOCKED_BIT) != 0) {
            return false;
        }
        
        // Try to set locked bit
        u64 new_state = current | LOCKED_BIT;
        u64 expected = current;
        return atomic_compare_exchange(
            &tagged_ptr_,
            &expected,
            new_state,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        );
    }
    
    // Unlock
    void unlock() {
        while (true) {
            u64 current = atomic_load(&tagged_ptr_, MemoryOrder::relaxed);
            
            // Check if waiters present
            if ((current & WAITERS_BIT) != 0) {
                // Clear locked bit, leaving WAITERS_BIT intact
                u64 new_state = current & ~LOCKED_BIT;
                u64 expected = current;
                if (atomic_compare_exchange(
                    &tagged_ptr_,
                    &expected,
                    new_state,
                    MemoryOrder::release,
                    MemoryOrder::relaxed
                )) {
                    // Successfully released lock.
                    // Under bucket lock, unpark one waiter and clear WAITERS_BIT if no more waiters remain.
                    get_parking_lot().unpark_one(this, [this](ParkingLot::UnparkResult res) {
                        if (res != ParkingLot::UnparkResult::HasMoreWaiters) {
                            atomic_fetch_and(&tagged_ptr_, ~WAITERS_BIT, MemoryOrder::relaxed);
                        }
                    });
                    return;
                }
                // CAS failed, retry
                continue;
            }
            
            // No waiters, just clear locked bit
            u64 new_state = current & ~LOCKED_BIT;
            u64 expected = current;
            if (atomic_compare_exchange(
                &tagged_ptr_,
                &expected,
                new_state,
                MemoryOrder::release,
                MemoryOrder::relaxed
            )) {
                // Successfully unlocked
                return;
            }
            // CAS failed, retry
        }
    }
    
    // Atomic pointer swap (exchange)
    // Returns old pointer (with direct bitmasking)
    T* exchange(T* new_ptr) {
        PSYNC_ASSERT((reinterpret_cast<u64>(new_ptr) & FLAG_MASK) == 0);
        
        u64 new_tagged = reinterpret_cast<u64>(new_ptr);
        u64 old_tagged = atomic_exchange(&tagged_ptr_, new_tagged, MemoryOrder::acq_rel);
        return reinterpret_cast<T*>(old_tagged & POINTER_MASK);
    }
    
    // Atomic compare-and-swap pointer
    // Returns true if successful (with direct bitmasking)
    bool compare_exchange(T* expected, T* desired) {
        PSYNC_ASSERT((reinterpret_cast<u64>(expected) & FLAG_MASK) == 0);
        PSYNC_ASSERT((reinterpret_cast<u64>(desired) & FLAG_MASK) == 0);
        
        u64 expected_tagged = reinterpret_cast<u64>(expected);
        u64 desired_tagged = reinterpret_cast<u64>(desired);
        
        return atomic_compare_exchange(
            &tagged_ptr_,
            &expected_tagged,
            desired_tagged,
            MemoryOrder::acq_rel,
            MemoryOrder::relaxed
        );
    }
    
    // Dereference (atomically get pointer)
    T* operator->() const {
        return get();
    }
    
    // Dereference (atomically get pointer)
    T& operator*() const {
        return *get();
    }
    
private:
    // Contended lock acquisition with parking
    void lock_contended() {
        // Thread-local waiter node (stack allocated)
        ParkingLot::WaiterNode node;
        
        while (true) {
            // Adaptive spin
            for (usize i = 0; i < SPIN_INITIAL; ++i) {
                cpu_pause();
                
                // Try to acquire lock
                u64 current = atomic_load(&tagged_ptr_, MemoryOrder::relaxed);
                if ((current & LOCKED_BIT) == 0) {
                    u64 new_state = current | LOCKED_BIT;
                    u64 expected = current;
                    if (atomic_compare_exchange(
                        &tagged_ptr_,
                        &expected,
                        new_state,
                        MemoryOrder::acquire,
                        MemoryOrder::relaxed
                    )) {
                        // Successfully acquired lock after spinning
                        return;
                    }
                }
            }
            
            // Set waiters flag and park
            while (true) {
                u64 current = atomic_load(&tagged_ptr_, MemoryOrder::relaxed);
                
                // If lock is unlocked, try to acquire it directly
                if ((current & LOCKED_BIT) == 0) {
                    u64 new_state = current | LOCKED_BIT;
                    u64 expected = current;
                    if (atomic_compare_exchange(
                        &tagged_ptr_,
                        &expected,
                        new_state,
                        MemoryOrder::acquire,
                        MemoryOrder::relaxed
                    )) {
                        // Successfully acquired lock
                        return;
                    }
                    // CAS failed, retry
                    continue;
                }
                
                // Park in parking lot with atomic validation & WAITERS_BIT setting under bucket lock
                get_parking_lot().park(this, &node, [this]() {
                    u64 cur = atomic_load(&tagged_ptr_, MemoryOrder::relaxed);
                    if ((cur & LOCKED_BIT) == 0) {
                        return false; // Already unlocked, do not park
                    }
                    if ((cur & WAITERS_BIT) == 0) {
                        atomic_fetch_or(&tagged_ptr_, WAITERS_BIT, MemoryOrder::relaxed);
                    }
                    return true;
                });
                break;
            }
            
            // After wake or validation abort, retry acquisition
        }
    }
    
    volatile u64 tagged_ptr_;
};

// ============================================================================
// TAGGED LOCK POINTER GUARD (RAII)
// ============================================================================

template<typename T>
class TaggedLockGuard {
public:
    explicit TaggedLockGuard(TaggedLockPtr<T>& ptr) : ptr_(ptr) {
        ptr_.lock();
    }
    
    ~TaggedLockGuard() {
        ptr_.unlock();
    }
    
    // Disable copy and move
    TaggedLockGuard(const TaggedLockGuard&) = delete;
    TaggedLockGuard& operator=(const TaggedLockGuard&) = delete;
    
    // Access to underlying pointer
    T* get() const {
        return ptr_.get();
    }
    
    T* operator->() const {
        return ptr_.get();
    }
    
    T& operator*() const {
        return *ptr_.get();
    }
    
private:
    TaggedLockPtr<T>& ptr_;
};

} // namespace psync