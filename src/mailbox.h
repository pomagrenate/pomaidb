// mailbox.h — Lock-free MPSC ring buffer for PomaiDB VectorRuntime dispatch.
// Backed by psync::MPSCQueue.
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <thread>
#include <psync/psync.h>

namespace pomai::core {

/**
 * BoundedMpscQueue: Lock-free MPSC ring buffer backed by psync::MPSCQueue.
 * Preserves PomaiDB's CamelCase API for seamless compatibility.
 */
template <class T>
class BoundedMpscQueue {
public:
    explicit BoundedMpscQueue(std::size_t capacity)
        : queue_(capacity) {}

    BoundedMpscQueue(const BoundedMpscQueue&) = delete;
    BoundedMpscQueue& operator=(const BoundedMpscQueue&) = delete;

    bool TryPush(T&& v) { return queue_.try_push(std::move(v)); }
    bool TryPush(const T& v) { return queue_.try_push(v); }
    bool PushBlocking(T&& v) { return queue_.push_blocking(std::move(v)); }
    bool PushBlocking(const T& v) { return queue_.push_blocking(v); }

    std::optional<T> PopBlocking() { return queue_.pop_blocking(); }
    std::optional<T> TryPop() { return queue_.try_pop(); }

    template <typename Rep, typename Period>
    std::optional<T> PopFor(std::chrono::duration<Rep, Period> timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        psync::usize backoff = 1;
        while (true) {
            auto v = TryPop();
            if (v.has_value()) return v;
            if (queue_.is_closed() && queue_.size_approx() == 0) return std::nullopt;
            if (std::chrono::steady_clock::now() >= deadline) return std::nullopt;
            for (psync::usize i = 0; i < backoff; ++i) psync::cpu_pause();
            if (backoff < 64) backoff <<= 1;
            else std::this_thread::yield();
        }
    }

    void Close() { queue_.close(); }
    std::size_t Size() const noexcept { return queue_.size_approx(); }
    bool Closed() const noexcept { return queue_.is_closed(); }
    std::size_t Capacity() const noexcept { return queue_.capacity(); }

private:
    psync::MPSCQueue<T> queue_;
};

} // namespace pomai::core
