#pragma once

// psync_condvar.h
// Zero-libc condition variable with thundering herd mitigation
// Target: Linux x86-64
// Compilers: GCC, Clang

#include "psync_platform.h"

#include <chrono>

namespace psync {

// Condition variable status for timed waits
enum class cv_status {
    no_timeout,
    timeout
};

// ============================================================================
// CONDITION VARIABLE
// ============================================================================
// Zero-libc, futex-backed condition variable for both Linux and Windows.
// Uses sequence/generation counter to eliminate lost wakeups completely.
// ============================================================================

class ConditionVariable {
public:
    // Initialize condition variable
    constexpr ConditionVariable() : seq_(0) {}
    
    // Disable copy and move
    ConditionVariable(const ConditionVariable&) = delete;
    ConditionVariable& operator=(const ConditionVariable&) = delete;
    
    // Wait on condition variable with predicate using any Lock type (e.g. UniqueLock<Mutex>)
    template<typename LockType, typename Predicate>
    void wait(LockType& lock, Predicate pred) {
        while (!pred()) {
            wait(lock);
        }
    }

    // Wait on condition variable using any Lock type (e.g. UniqueLock<Mutex>)
    template<typename LockType>
    void wait(LockType& lock) {
        u32 seq = atomic_load(&seq_, MemoryOrder::relaxed);
        lock.unlock();
        futex::wait(&seq_, seq);
        lock.lock();
    }

    // Wait on condition variable with predicate for raw Mutex
    template<typename Predicate>
    void wait(Mutex& mutex, Predicate pred) {
        while (!pred()) {
            wait(mutex);
        }
    }
    
    // Wait on condition variable for raw Mutex
    void wait(Mutex& mutex) {
        u32 seq = atomic_load(&seq_, MemoryOrder::relaxed);
        mutex.unlock();
        futex::wait(&seq_, seq);
        mutex.lock();
    }

    // Timed wait with relative duration
    template<typename LockType, typename Rep, typename Period>
    cv_status wait_for(LockType& lock, const std::chrono::duration<Rep, Period>& rel_time) {
        u32 seq = atomic_load(&seq_, MemoryOrder::relaxed);
        lock.unlock();
        u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(rel_time).count());
        futex::wait_for(&seq_, seq, ns);
        lock.lock();
        return (atomic_load(&seq_, MemoryOrder::relaxed) == seq) ? cv_status::timeout : cv_status::no_timeout;
    }

    // Timed wait with predicate and relative duration
    template<typename LockType, typename Rep, typename Period, typename Predicate>
    bool wait_for(LockType& lock, const std::chrono::duration<Rep, Period>& rel_time, Predicate pred) {
        auto deadline = std::chrono::steady_clock::now() + rel_time;
        while (!pred()) {
            auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return pred();
            }
            wait_for(lock, deadline - now);
        }
        return true;
    }

    // Timed wait with absolute time point
    template<typename LockType, typename Clock, typename Duration>
    cv_status wait_until(LockType& lock, const std::chrono::time_point<Clock, Duration>& timeout_time) {
        auto now = Clock::now();
        if (now >= timeout_time) {
            return cv_status::timeout;
        }
        return wait_for(lock, timeout_time - now);
    }

    // Timed wait with predicate and absolute time point
    template<typename LockType, typename Clock, typename Duration, typename Predicate>
    bool wait_until(LockType& lock, const std::chrono::time_point<Clock, Duration>& timeout_time, Predicate pred) {
        while (!pred()) {
            if (Clock::now() >= timeout_time) {
                return pred();
            }
            wait_until(lock, timeout_time);
        }
        return true;
    }
    
    // Wake one waiting thread
    void signal() {
        atomic_fetch_add(&seq_, 1u, MemoryOrder::release);
        futex::wake(&seq_, 1);
    }
    
    // Standard alias for signal()
    void notify_one() {
        signal();
    }
    
    // Wake all waiting threads
    void broadcast(Mutex& mutex) {
#ifdef PSYNC_PLATFORM_LINUX
        u32 old_seq = atomic_fetch_add(&seq_, 1u, MemoryOrder::release);
        volatile u32* mutex_state = mutex.get_state_address();
        futex::requeue(&seq_, mutex_state, 0x7fffffff, old_seq);
#else
        (void)mutex;
        broadcast();
#endif
    }
    
    // Wake all waiting threads (simplified version without mutex)
    void broadcast() {
        atomic_fetch_add(&seq_, 1u, MemoryOrder::release);
        futex::wake_all(&seq_);
    }

    // Standard alias for broadcast()
    void notify_all() {
        broadcast();
    }

    void notify_all(Mutex& mutex) {
        broadcast(mutex);
    }
    
private:
    volatile u32 seq_;
};

// ============================================================================
// CONDITION VARIABLE GUARD (RAII)
// ============================================================================

class ConditionVariableGuard {
public:
    ConditionVariableGuard(ConditionVariable& cv, Mutex& mutex) 
        : cv_(cv), mutex_(mutex) {
        cv_.wait(mutex_);
    }
    
    template<typename Predicate>
    ConditionVariableGuard(ConditionVariable& cv, Mutex& mutex, Predicate pred) 
        : cv_(cv), mutex_(mutex) {
        cv_.wait(mutex, pred);
    }
    
    ~ConditionVariableGuard() {
        // Mutex is automatically re-acquired by wait()
    }
    
    // Disable copy and move
    ConditionVariableGuard(const ConditionVariableGuard&) = delete;
    ConditionVariableGuard& operator=(const ConditionVariableGuard&) = delete;
    
private:
    ConditionVariable& cv_;
    Mutex& mutex_;
};

} // namespace psync