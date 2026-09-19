#pragma once

// ptask_deque.h
// Lock-free Chase-Lev Work-Stealing Deque (Single-Producer, Multi-Consumer).
// Based on the Chase-Lev algorithm with dynamic circular array expansion.
// Part of ptask - High-performance work-stealing task scheduler.

#include "ptask_types.h"
#include <atomic>
#include <vector>
#include <optional>
#include <cassert>

namespace ptask {

template <typename T>
class WorkStealingDeque {
private:
    class CircularArray {
    public:
        explicit CircularArray(int64_t capacity)
            : capacity_(capacity),
              mask_(capacity - 1),
              storage_(new std::atomic<T>[capacity]) {
            assert((capacity & (capacity - 1)) == 0 && "Capacity must be a power of 2");
            for (int64_t i = 0; i < capacity_; ++i) {
                storage_[i].store(T{}, std::memory_order_relaxed);
            }
        }

        ~CircularArray() {
            delete[] storage_;
        }

        int64_t capacity() const noexcept {
            return capacity_;
        }

        void put(int64_t index, T item) noexcept {
            storage_[index & mask_].store(item, std::memory_order_relaxed);
        }

        T get(int64_t index) const noexcept {
            return storage_[index & mask_].load(std::memory_order_relaxed);
        }

        CircularArray* grow(int64_t bottom, int64_t top) {
            CircularArray* new_array = new CircularArray(capacity_ * 2);
            for (int64_t i = top; i < bottom; ++i) {
                new_array->put(i, get(i));
            }
            return new_array;
        }

    private:
        int64_t capacity_;
        int64_t mask_;
        std::atomic<T>* storage_;
    };

    alignas(64) std::atomic<int64_t> top_{0};
    alignas(64) std::atomic<int64_t> bottom_{0};
    alignas(64) std::atomic<CircularArray*> array_{nullptr};
    std::vector<std::unique_ptr<CircularArray>> old_arrays_;

public:
    explicit WorkStealingDeque(int64_t initial_capacity = 1024) {
        CircularArray* arr = new CircularArray(initial_capacity);
        array_.store(arr, std::memory_order_relaxed);
    }

    ~WorkStealingDeque() {
        CircularArray* current = array_.load(std::memory_order_relaxed);
        delete current;
    }

    // Disable copy and move
    WorkStealingDeque(const WorkStealingDeque&) = delete;
    WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;

    // Push item to the bottom of the deque (Owner thread only)
    void push(T item) {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_acquire);
        CircularArray* a = array_.load(std::memory_order_relaxed);

        if (b - t >= a->capacity() - 1) {
            CircularArray* new_a = a->grow(b, t);
            old_arrays_.emplace_back(a);
            a = new_a;
            array_.store(a, std::memory_order_release);
        }

        a->put(b, item);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
    }

    // Pop item from the bottom of the deque (Owner thread only)
    std::optional<T> pop() {
        int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        CircularArray* a = array_.load(std::memory_order_relaxed);
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t t = top_.load(std::memory_order_relaxed);

        if (t <= b) {
            // Queue is not empty
            T item = a->get(b);
            if (t == b) {
                // Last item, compete with concurrent stealers
                if (!top_.compare_exchange_strong(t, t + 1, 
                                                  std::memory_order_seq_cst, 
                                                  std::memory_order_relaxed)) {
                    // Lost race to a stealer
                    bottom_.store(b + 1, std::memory_order_relaxed);
                    return std::nullopt;
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
                return item;
            }
            return item;
        } else {
            // Queue was already empty
            bottom_.store(b + 1, std::memory_order_relaxed);
            return std::nullopt;
        }
    }

    // Steal item from the top of the deque (Concurrent stealer threads)
    std::optional<T> steal() {
        int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t b = bottom_.load(std::memory_order_acquire);

        if (t < b) {
            CircularArray* a = array_.load(std::memory_order_acquire);
            T item = a->get(t);

            if (top_.compare_exchange_strong(t, t + 1, 
                                              std::memory_order_seq_cst, 
                                              std::memory_order_relaxed)) {
                return item;
            }
        }
        return std::nullopt;
    }

    // Check if deque is approximately empty
    bool empty() const noexcept {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        return b <= t;
    }

    // Approximate size
    int64_t size() const noexcept {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        return b > t ? (b - t) : 0;
    }
};

} // namespace ptask
