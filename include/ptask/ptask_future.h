#pragma once

// ptask_future.h
// Lightweight cooperative task futures and promises.
// Enables non-blocking, work-stealing waits without thread starvation.
// Part of ptask - High-performance work-stealing task scheduler.

#include "ptask_types.h"
#include <atomic>
#include <memory>
#include <exception>
#include <cassert>

namespace ptask {

class ThreadPool;

// ============================================================================
// SHARED STATE FOR FUTURES
// ============================================================================

template <typename T>
struct FutureState {
    std::atomic<bool> ready{false};
    std::exception_ptr exception{nullptr};
    alignas(alignof(T)) std::byte storage[sizeof(T)];

    ~FutureState() {
        if (ready.load(std::memory_order_acquire) && !exception) {
            reinterpret_cast<T*>(storage)->~T();
        }
    }

    void set_value(T&& val) {
        new (storage) T(std::move(val));
        ready.store(true, std::memory_order_release);
    }

    void set_value(const T& val) {
        new (storage) T(val);
        ready.store(true, std::memory_order_release);
    }

    void set_exception(std::exception_ptr e) {
        exception = e;
        ready.store(true, std::memory_order_release);
    }

    T& get() {
        if (exception) {
            std::rethrow_exception(exception);
        }
        return *reinterpret_cast<T*>(storage);
    }
};

// Specialization for void
template <>
struct FutureState<void> {
    std::atomic<bool> ready{false};
    std::exception_ptr exception{nullptr};

    void set_value() {
        ready.store(true, std::memory_order_release);
    }

    void set_exception(std::exception_ptr e) {
        exception = e;
        ready.store(true, std::memory_order_release);
    }

    void get() {
        if (exception) {
            std::rethrow_exception(exception);
        }
    }
};

// ============================================================================
// TASK FUTURE
// ============================================================================

template <typename T>
class TaskFuture {
public:
    TaskFuture() noexcept : state_(nullptr), pool_(nullptr) {}

    explicit TaskFuture(std::shared_ptr<FutureState<T>> state, ThreadPool* pool = nullptr)
        : state_(std::move(state)), pool_(pool) {}

    bool is_ready() const noexcept {
        return state_ && state_->ready.load(std::memory_order_acquire);
    }

    // Cooperative wait: if inside thread pool, assists with work instead of blocking
    void wait() const;

    // Retrieve value, waiting cooperatively until available
    decltype(auto) get() {
        assert(state_ && "Attempted to get() on uninitialized TaskFuture");
        wait();
        return state_->get();
    }

private:
    std::shared_ptr<FutureState<T>> state_;
    ThreadPool* pool_;
};

} // namespace ptask
