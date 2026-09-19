#pragma once

// psync_mpsc_intrusive.h
// Intrusive, zero-allocation lock-free MPSC queue
// Target: Linux / macOS / Windows across x86-64, ARM64, RISC-V

#include "psync_platform.h"
#include "psync_cache_padded.h"
#include <atomic>

namespace psync {

/**
 * MPSCQueueEntry: Base class for intrusive items queued in MPSCIntrusiveQueue.
 * The entry pointer is stored within the object itself, eliminating dynamic
 * node allocations on push.
 */
struct MPSCQueueEntry {
    std::atomic<MPSCQueueEntry*> next{nullptr};
};

/**
 * MPSCIntrusiveQueue<T>: Zero-allocation, intrusive lock-free queue.
 * Optimal for thread-per-core architectures and task dispatchers where multiple
 * producer threads enqueue tasks to a single owner thread.
 */
template <typename T>
class MPSCIntrusiveQueue {
public:
    MPSCIntrusiveQueue()
        : stub_(),
          head_(&stub_),
          tail_(&stub_)
    {
        stub_.next.store(nullptr, std::memory_order_relaxed);
    }

    MPSCIntrusiveQueue(const MPSCIntrusiveQueue&) = delete;
    MPSCIntrusiveQueue& operator=(const MPSCIntrusiveQueue&) = delete;

    /**
     * Push an entry onto the queue. Thread-safe for multiple concurrent producers.
     * Uses atomic exchange for O(1) lock-free insertion.
     */
    void push(MPSCQueueEntry* entry) noexcept {
        entry->next.store(nullptr, std::memory_order_relaxed);
        MPSCQueueEntry* prev = head_.value.exchange(entry, std::memory_order_acq_rel);
        prev->next.store(entry, std::memory_order_release);
    }

    /**
     * Pop an entry from the queue. ONLY safe for a single consumer thread.
     * Returns nullptr if the queue is empty.
     */
    T* pop() noexcept {
        MPSCQueueEntry* tail = tail_.value;
        MPSCQueueEntry* next = tail->next.load(std::memory_order_acquire);

        if (tail == &stub_) {
            if (nullptr == next) return nullptr;
            tail_.value = next;
            tail = next;
            next = next->next.load(std::memory_order_acquire);
        }

        if (next) {
            tail_.value = next;
            return static_cast<T*>(tail);
        }

        MPSCQueueEntry* head = head_.value.load(std::memory_order_acquire);
        if (tail != head) {
            // Producer in the middle of exchange -> store
            return nullptr;
        }

        push(&stub_);
        next = tail->next.load(std::memory_order_acquire);
        if (next) {
            tail_.value = next;
            return static_cast<T*>(tail);
        }

        return nullptr;
    }

private:
    MPSCQueueEntry stub_;
    CachePadded<std::atomic<MPSCQueueEntry*>> head_;
    CachePadded<MPSCQueueEntry*> tail_;
};

} // namespace psync
