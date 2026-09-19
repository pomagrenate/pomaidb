#pragma once

// psync_spsc_queue.h
// Wait-free Single-Producer Single-Consumer (SPSC) ring buffer
// Target: Linux / macOS / Windows across x86-64, ARM64, RISC-V

#include "psync_platform.h"
#include "psync_cache_padded.h"
#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

namespace psync {

/**
 * SPSCQueue<T>: Wait-free Single-Producer Single-Consumer bounded ring buffer.
 * Achieves peak throughput with zero atomic read-modify-write (CAS) operations.
 * Uses cached head and tail pointers to minimize cacheline bouncing between cores.
 */
template <typename T>
class SPSCQueue {
public:
    explicit SPSCQueue(usize capacity)
        : mask_(round_up_pow2(capacity < 2 ? 2 : capacity) - 1),
          buffer_(static_cast<T*>(operator new[]((mask_ + 1) * sizeof(T))))
    {
        head_.value.store(0, std::memory_order_relaxed);
        tail_.value.store(0, std::memory_order_relaxed);
        tail_cached_.value = 0;
        head_cached_.value = 0;
    }

    ~SPSCQueue() {
        T dummy;
        while (try_pop(dummy)) {}
        operator delete[](buffer_);
    }

    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    /**
     * Producer: push item. Wait-free.
     */
    bool try_push(T&& value) {
        usize head = head_.value.load(std::memory_order_relaxed);
        if (head - tail_cached_.value > mask_) {
            tail_cached_.value = tail_.value.load(std::memory_order_acquire);
            if (head - tail_cached_.value > mask_) {
                return false; // Full
            }
        }

        new (&buffer_[head & mask_]) T(std::move(value));
        head_.value.store(head + 1, std::memory_order_release);
        return true;
    }

    bool try_push(const T& value) {
        T copy = value;
        return try_push(std::move(copy));
    }

    /**
     * Consumer: pop item into out parameter. Wait-free.
     */
    bool try_pop(T& out) {
        usize tail = tail_.value.load(std::memory_order_relaxed);
        if (tail == head_cached_.value) {
            head_cached_.value = head_.value.load(std::memory_order_acquire);
            if (tail == head_cached_.value) {
                return false; // Empty
            }
        }

        T* slot = &buffer_[tail & mask_];
        out = std::move(*slot);
        slot->~T();
        tail_.value.store(tail + 1, std::memory_order_release);
        return true;
    }

    /**
     * Consumer: pop item returning std::optional.
     */
    std::optional<T> try_pop() {
        T val;
        if (try_pop(val)) {
            return std::move(val);
        }
        return std::nullopt;
    }

    bool empty() const noexcept {
        return tail_.value.load(std::memory_order_relaxed) ==
               head_.value.load(std::memory_order_relaxed);
    }

    usize capacity() const noexcept {
        return mask_ + 1;
    }

private:
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
    T* const buffer_;

    CachePadded<std::atomic<usize>> head_;
    CachePadded<usize> tail_cached_;

    CachePadded<std::atomic<usize>> tail_;
    CachePadded<usize> head_cached_;
};

} // namespace psync
