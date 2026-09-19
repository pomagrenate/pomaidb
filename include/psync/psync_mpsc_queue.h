#pragma once

// psync_mpsc_queue.h
// Lock-free Multi-Producer Single-Consumer (MPSC) bounded ring buffer
// Inspired by Dmitry Vyukov's bounded MPMC/MPSC queue
// Target: Linux / macOS / Windows across x86-64, ARM64, RISC-V

#include "psync_platform.h"
#include "psync_cache_padded.h"
#include <atomic>
#include <cstddef>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace psync {

/**
 * MPSCQueue<T>: High-throughput lock-free bounded ring buffer.
 * Multiple producer threads can enqueue concurrently via atomic compare-and-swap
 * on the head sequence. The single consumer thread dequeues without taking any lock.
 */
template <typename T>
class MPSCQueue {
public:
    explicit MPSCQueue(usize capacity)
        : mask_(round_up_pow2(capacity < 2 ? 2 : capacity) - 1),
          buffer_(new Slot[mask_ + 1])
    {
        for (usize i = 0; i <= mask_; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
        head_.value.store(0, std::memory_order_relaxed);
        tail_.value = 0;
        closed_.store(false, std::memory_order_relaxed);
        size_approx_.store(0, std::memory_order_relaxed);
    }

    ~MPSCQueue() {
        delete[] buffer_;
    }

    MPSCQueue(const MPSCQueue&) = delete;
    MPSCQueue& operator=(const MPSCQueue&) = delete;

    /**
     * Producer path (thread-safe for multiple concurrent producers).
     * Returns true if successfully enqueued, false if queue is full or closed.
     */
    bool try_push(T&& value) {
        usize head = head_.value.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = buffer_[head & mask_];
            usize seq = slot.sequence.load(std::memory_order_acquire);
            isize diff = static_cast<isize>(seq) - static_cast<isize>(head);

            if (diff == 0) {
                // Slot is ready for this head generation
                if (head_.value.compare_exchange_weak(
                    head, head + 1,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed
                )) {
                    if (closed_.load(std::memory_order_relaxed)) {
                        // Revert turn on closure
                        slot.sequence.store(head, std::memory_order_release);
                        return false;
                    }
                    slot.data = std::move(value);
                    slot.sequence.store(head + 1, std::memory_order_release);
                    size_approx_.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
            } else if (diff < 0) {
                // Buffer is full
                return false;
            } else {
                head = head_.value.load(std::memory_order_relaxed);
            }
        }
    }

    bool try_push(const T& value) {
        T copy = value;
        return try_push(std::move(copy));
    }

    /**
     * Blocking push with exponential backoff and yielding.
     */
    bool push_blocking(T&& value) {
        usize backoff = 1;
        while (!closed_.load(std::memory_order_relaxed)) {
            if (try_push(std::move(value))) return true;
            for (usize i = 0; i < backoff; ++i) cpu_pause();
            if (backoff < 64) backoff <<= 1;
            else std::this_thread::yield();
        }
        return false;
    }

    /**
     * Consumer path (single-consumer only).
     * Returns the popped value if available, or std::nullopt if empty.
     */
    std::optional<T> try_pop() {
        usize tail = tail_.value;
        Slot& slot = buffer_[tail & mask_];
        usize seq = slot.sequence.load(std::memory_order_acquire);
        isize diff = static_cast<isize>(seq) - static_cast<isize>(tail + 1);

        if (diff == 0) {
            // Slot has new data
            tail_.value = tail + 1;
            T result = std::move(slot.data);
            slot.sequence.store(tail + mask_ + 1, std::memory_order_release);
            size_approx_.fetch_sub(1, std::memory_order_relaxed);
            return result;
        }
        return std::nullopt;
    }

    /**
     * Blocking pop with adaptive pause and yield.
     */
    std::optional<T> pop_blocking() {
        usize backoff = 1;
        while (true) {
            auto val = try_pop();
            if (val.has_value()) return val;
            if (closed_.load(std::memory_order_relaxed)) {
                return try_pop();
            }
            for (usize i = 0; i < backoff; ++i) cpu_pause();
            if (backoff < 64) backoff <<= 1;
            else std::this_thread::yield();
        }
    }

    void close() noexcept {
        closed_.store(true, std::memory_order_release);
    }

    bool is_closed() const noexcept {
        return closed_.load(std::memory_order_acquire);
    }

    usize size_approx() const noexcept {
        return size_approx_.load(std::memory_order_relaxed);
    }

    usize capacity() const noexcept {
        return mask_ + 1;
    }

private:
    struct Slot {
        std::atomic<usize> sequence;
        T data;
    };

    static usize round_up_pow2(usize v) {
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
#if INTPTR_MAX == INT64_MAX
        v |= v >> 32;
#endif
        return v + 1;
    }

    const usize mask_;
    Slot* const buffer_;

    CachePadded<std::atomic<usize>> head_;
    CachePadded<usize> tail_;
    std::atomic<bool> closed_{false};
    std::atomic<usize> size_approx_{0};
};

} // namespace psync
