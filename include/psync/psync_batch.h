#pragma once

// psync_batch.h
// Batch synchronization for amortizing lock overhead
// Target: Linux x86-64 / Windows
// Compilers: GCC, Clang, MSVC

#include "psync_platform.h"
#include "psync_mutex.h"
#include "psync_microlock.h"
#include "psync_shared.h"

namespace psync {

// ============================================================================
// BATCH GUARD (RAII for batched operations)
// ============================================================================

template<typename Lock>
class BatchGuard {
public:
    explicit BatchGuard(Lock& lock) : lock_(lock) {
        lock_.lock();
    }
    
    ~BatchGuard() {
        lock_.unlock();
    }
    
    // Disable copy and move
    BatchGuard(const BatchGuard&) = delete;
    BatchGuard& operator=(const BatchGuard&) = delete;
    
private:
    Lock& lock_;
};

// ============================================================================
// SHARED BATCH GUARD (RAII for batched shared operations)
// ============================================================================

class SharedBatchGuard {
public:
    explicit SharedBatchGuard(SharedMutex& mutex) : mutex_(mutex) {
        mutex_.lock_shared();
    }
    
    ~SharedBatchGuard() {
        mutex_.unlock_shared();
    }
    
    // Disable copy and move
    SharedBatchGuard(const SharedBatchGuard&) = delete;
    SharedBatchGuard& operator=(const SharedBatchGuard&) = delete;
    
private:
    SharedMutex& mutex_;
};

// ============================================================================
// EXCLUSIVE BATCH GUARD (RAII for batched exclusive operations)
// ============================================================================

class ExclusiveBatchGuard {
public:
    explicit ExclusiveBatchGuard(SharedMutex& mutex) : mutex_(mutex) {
        mutex_.lock();
    }
    
    ~ExclusiveBatchGuard() {
        mutex_.unlock();
    }
    
    // Disable copy and move
    ExclusiveBatchGuard(const ExclusiveBatchGuard&) = delete;
    ExclusiveBatchGuard& operator=(const ExclusiveBatchGuard&) = delete;
    
private:
    SharedMutex& mutex_;
};

// ============================================================================
// BATCH OPERATION HELPERS
// ============================================================================

// Execute a batch of operations under a lock
// Usage: batch_execute(mutex, [&]() { /* multiple operations */ });
template<typename Lock, typename Func>
void batch_execute(Lock& lock, Func&& func) {
    BatchGuard<Lock> guard(lock);
    func();
}

// Execute a batch of operations under a shared lock
// Usage: batch_execute_shared(mutex, [&]() { /* multiple read operations */ });
template<typename Func>
void batch_execute_shared(SharedMutex& mutex, Func&& func) {
    SharedLockGuard guard(mutex);
    func();
}

// Execute a batch of operations under an exclusive lock
// Usage: batch_execute_exclusive(mutex, [&]() { /* multiple write operations */ });
template<typename Func>
void batch_execute_exclusive(SharedMutex& mutex, Func&& func) {
    UniqueLockGuard guard(mutex);
    func();
}

// ============================================================================
// BATCH SIZE TRACKING (for performance analysis)
// ============================================================================

class BatchCounter {
public:
    constexpr BatchCounter() : count_(0) {}
    
    void increment() {
        atomic_fetch_add(&count_, static_cast<u32>(1), MemoryOrder::relaxed);
    }
    
    u32 get() const {
        return atomic_load(&count_, MemoryOrder::relaxed);
    }
    
    void reset() {
        atomic_store(&count_, static_cast<u32>(0), MemoryOrder::relaxed);
    }
    
private:
    volatile u32 count_;
};

// ============================================================================
// BATCH OPERATION COUNTER
// ============================================================================

// Counter for tracking batch sizes
// Can be used to measure batch effectiveness
class BatchTracker {
public:
    constexpr BatchTracker() : total_operations_(0), total_batches_(0) {}
    
    void record_batch(usize operations) {
        atomic_fetch_add(&total_operations_, static_cast<u32>(operations), MemoryOrder::relaxed);
        atomic_fetch_add(&total_batches_, static_cast<u32>(1), MemoryOrder::relaxed);
    }
    
    u32 get_total_operations() const {
        return atomic_load(&total_operations_, MemoryOrder::relaxed);
    }
    
    u32 get_total_batches() const {
        return atomic_load(&total_batches_, MemoryOrder::relaxed);
    }
    
    double get_average_batch_size() const {
        u32 ops = get_total_operations();
        u32 batches = get_total_batches();
        if (batches == 0) return 0.0;
        return static_cast<double>(ops) / static_cast<double>(batches);
    }
    
    void reset() {
        atomic_store(&total_operations_, static_cast<u32>(0), MemoryOrder::relaxed);
        atomic_store(&total_batches_, static_cast<u32>(0), MemoryOrder::relaxed);
    }
    
private:
    volatile u32 total_operations_;
    volatile u32 total_batches_;
};

} // namespace psync
