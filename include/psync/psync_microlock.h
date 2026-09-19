#pragma once

// psync_microlock.h
// 1-byte microlock with external parking lot
// Target: Linux x86-64 / Windows
// Compilers: GCC, Clang, MSVC

#include "psync_platform.h"

namespace psync {

// ============================================================================
// MICROLOCK STATE LAYOUT (8 bits)
// ============================================================================
// Bits 7-1: Reserved (future use)
// Bit 0:    LOCKED flag (1 = locked, 0 = unlocked)
// ============================================================================

// ============================================================================
// PARKING LOT
// ============================================================================
// Global hash-based parking lot for microlocks
// Lock address is hashed to a bucket
// Each bucket contains a waiter queue
// ============================================================================

class ParkingLot {
public:
    // Waiter node (intrusive list)
    struct WaiterNode {
        const void* key;
        volatile u32 wake_flag;
        WaiterNode* next;
        
        WaiterNode() : key(nullptr), wake_flag(0), next(nullptr) {}
    };
    
    // Bucket containing waiter queue
    struct Bucket {
        volatile u32 lock_state;  // Internal lock for bucket
        WaiterNode* head;
        WaiterNode* tail;
        
        Bucket() : lock_state(0), head(nullptr), tail(nullptr) {}
    };
    
    // Number of buckets (power of 2 for efficient hashing)
    static constexpr usize NUM_BUCKETS = 64;
    
    // Result of unpark_one operation
    enum class UnparkResult {
        None,             // No waiter found for this key
        NoMoreWaiters,    // Successfully unparked one, no more waiters for this key
        HasMoreWaiters    // Successfully unparked one, more waiters still exist for this key
    };
    
    // Initialize parking lot (trivial initialization)
    ParkingLot() {
        for (usize i = 0; i < NUM_BUCKETS; ++i) {
            buckets_[i].lock_state = 0;
            buckets_[i].head = nullptr;
            buckets_[i].tail = nullptr;
        }
    }
    
    // Park a thread waiting on a lock address with validation callback
    // Returns true if parked and woken, false if validation failed
    template<typename ValidateFunc>
    bool park(const void* lock_address, WaiterNode* node, ValidateFunc&& validate) {
        usize bucket_index = hash_address(lock_address);
        Bucket* bucket = &buckets_[bucket_index];
        
        // Lock the bucket
        lock_bucket(bucket);
        
        // Validate condition while holding bucket lock to prevent lost wakeup
        if (!validate()) {
            unlock_bucket(bucket);
            return false;
        }
        
        // Add waiter to queue
        node->key = lock_address;
        node->wake_flag = 0;
        node->next = nullptr;
        
        if (bucket->tail) {
            bucket->tail->next = node;
        } else {
            bucket->head = node;
        }
        bucket->tail = node;
        
        // Unlock the bucket
        unlock_bucket(bucket);
        
        // Wait on wake flag
        while (atomic_load(&node->wake_flag, MemoryOrder::relaxed) == 0) {
            futex::wait(&node->wake_flag, 0);
        }
        
        return true;
    }
    
    // Default park without validation
    bool park(const void* lock_address, WaiterNode* node) {
        return park(lock_address, node, []() { return true; });
    }
    
    // Unpark one thread waiting on a lock address with callback under bucket lock
    template<typename Callback>
    UnparkResult unpark_one(const void* lock_address, Callback&& callback) {
        usize bucket_index = hash_address(lock_address);
        Bucket* bucket = &buckets_[bucket_index];
        
        lock_bucket(bucket);
        
        WaiterNode* prev = nullptr;
        WaiterNode* curr = bucket->head;
        WaiterNode* target = nullptr;
        
        while (curr) {
            if (curr->key == lock_address) {
                target = curr;
                if (prev) {
                    prev->next = curr->next;
                } else {
                    bucket->head = curr->next;
                }
                if (bucket->tail == curr) {
                    bucket->tail = prev;
                }
                break;
            }
            prev = curr;
            curr = curr->next;
        }
        
        if (!target) {
            callback(UnparkResult::None);
            unlock_bucket(bucket);
            return UnparkResult::None;
        }
        
        // Check if there are more waiters for this key
        bool has_more = false;
        WaiterNode* check = bucket->head;
        while (check) {
            if (check->key == lock_address) {
                has_more = true;
                break;
            }
            check = check->next;
        }
        
        UnparkResult result = has_more ? UnparkResult::HasMoreWaiters : UnparkResult::NoMoreWaiters;
        
        // Invoke callback under bucket lock before releasing and waking
        callback(result);
        
        // Set wake flag BEFORE unlocking bucket
        atomic_store(&target->wake_flag, static_cast<u32>(1), MemoryOrder::release);
        unlock_bucket(bucket);
        
        // Wake the waiter
        futex::wake(&target->wake_flag, 1);
        return result;
    }
    
    // Default unpark_one without callback
    UnparkResult unpark_one(const void* lock_address) {
        return unpark_one(lock_address, [](UnparkResult) {});
    }
    
    // Unpark all threads waiting on a specific lock address
    usize unpark_all(const void* lock_address) {
        usize bucket_index = hash_address(lock_address);
        Bucket* bucket = &buckets_[bucket_index];
        
        lock_bucket(bucket);
        
        WaiterNode* to_wake_head = nullptr;
        WaiterNode* to_wake_tail = nullptr;
        usize count = 0;
        
        WaiterNode* prev = nullptr;
        WaiterNode* curr = bucket->head;
        while (curr) {
            WaiterNode* next = curr->next;
            if (curr->key == lock_address) {
                if (prev) {
                    prev->next = next;
                } else {
                    bucket->head = next;
                }
                if (bucket->tail == curr) {
                    bucket->tail = prev;
                }
                
                curr->next = nullptr;
                if (to_wake_tail) {
                    to_wake_tail->next = curr;
                } else {
                    to_wake_head = curr;
                }
                to_wake_tail = curr;
                count++;
            } else {
                prev = curr;
            }
            curr = next;
        }
        
        unlock_bucket(bucket);
        
        // Wake all matching waiters outside bucket lock
        curr = to_wake_head;
        while (curr) {
            WaiterNode* next = curr->next;
            atomic_store(&curr->wake_flag, static_cast<u32>(1), MemoryOrder::release);
            futex::wake(&curr->wake_flag, 1);
            curr = next;
        }
        
        return count;
    }
    
private:
    // Hash function for lock address
    static usize hash_address(const void* address) {
        usize addr = reinterpret_cast<usize>(address);
        return (addr >> 3) & (NUM_BUCKETS - 1);
    }
    
    // Lock a bucket (spinlock)
    void lock_bucket(Bucket* bucket) {
        while (true) {
            u32 expected = 0;
            if (atomic_compare_exchange(
                &bucket->lock_state,
                &expected,
                static_cast<u32>(1),
                MemoryOrder::acquire,
                MemoryOrder::relaxed
            )) {
                return;
            }
            
            // Adaptive spin
            for (usize i = 0; i < SPIN_INITIAL; ++i) {
                cpu_pause();
            }
        }
    }
    
    // Unlock a bucket
    void unlock_bucket(Bucket* bucket) {
        atomic_store(&bucket->lock_state, static_cast<u32>(0), MemoryOrder::release);
    }
    
    Bucket buckets_[NUM_BUCKETS];
};

// Get global parking lot instance (function-local static for thread-safe initialization)
inline ParkingLot& get_parking_lot() {
    static ParkingLot instance;
    return instance;
}

// ============================================================================
// MICROLOCK
// ============================================================================

class MicroLock {
public:
    // State constants
    static constexpr u8 UNLOCKED = 0x00u;
    static constexpr u8 LOCKED = 0x01u;
    
    // Initialize microlock (trivial initialization)
    constexpr MicroLock() : state_(UNLOCKED) {}
    
    // Disable copy and move
    MicroLock(const MicroLock&) = delete;
    MicroLock& operator=(const MicroLock&) = delete;
    
    // Acquire the microlock
    void lock() {
        // Fast path: try to acquire uncontended lock
        u8 expected = UNLOCKED;
        if (atomic_compare_exchange(
            &state_,
            &expected,
            LOCKED,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        )) {
            return;
        }
        
        // Slow path: contended acquisition with parking
        lock_contended();
    }
    
    // Try to acquire the microlock without blocking
    bool try_lock() {
        u8 expected = UNLOCKED;
        return atomic_compare_exchange(
            &state_,
            &expected,
            LOCKED,
            MemoryOrder::acquire,
            MemoryOrder::relaxed
        );
    }
    
    // Release the microlock
    void unlock() {
        // Clear locked flag
        atomic_store(&state_, UNLOCKED, MemoryOrder::release);
        
        // Wake one waiter if any
        get_parking_lot().unpark_one(this);
    }
    
private:
    // Contended lock acquisition with parking
    void lock_contended() {
        ParkingLot::WaiterNode node;
        
        while (true) {
            // Adaptive spin
            for (usize i = 0; i < SPIN_INITIAL; ++i) {
                cpu_pause();
                
                u8 expected = UNLOCKED;
                if (atomic_compare_exchange(
                    &state_,
                    &expected,
                    LOCKED,
                    MemoryOrder::acquire,
                    MemoryOrder::relaxed
                )) {
                    return;
                }
            }
            
            // Park in parking lot with validation under bucket lock
            get_parking_lot().park(this, &node, [this]() {
                return atomic_load(&state_, MemoryOrder::relaxed) == LOCKED;
            });
            
            // After wake or if validation aborted, retry acquisition
            u8 expected = UNLOCKED;
            if (atomic_compare_exchange(
                &state_,
                &expected,
                LOCKED,
                MemoryOrder::acquire,
                MemoryOrder::relaxed
            )) {
                return;
            }
        }
    }
    
    volatile u8 state_;
};

// ============================================================================
// MICROLOCK GUARD (RAII)
// ============================================================================

class MicroLockGuard {
public:
    explicit MicroLockGuard(MicroLock& lock) : lock_(lock) {
        lock_.lock();
    }
    
    ~MicroLockGuard() {
        lock_.unlock();
    }
    
    // Disable copy and move
    MicroLockGuard(const MicroLockGuard&) = delete;
    MicroLockGuard& operator=(const MicroLockGuard&) = delete;
    
private:
    MicroLock& lock_;
};

} // namespace psync
