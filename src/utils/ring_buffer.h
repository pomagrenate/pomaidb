#pragma once
#include <vector>
#include <array>
#include <optional>
#include <stdexcept>
#include <new>
#include "palloc_compat.h"

namespace pomai::core::util {

    /**
     * A simple, fixed-capacity circular buffer (Ring Buffer).
     * Designed for zero-allocation performance in single-threaded environments.
     */
    template <typename T, std::size_t Capacity>
    class StaticRingBuffer {
    public:
        StaticRingBuffer() : head_(0), tail_(0), size_(0) {}

        bool push_back(T&& value) {
            if (size_ == Capacity) return false; // Full
            data_[tail_] = std::move(value);
            tail_ = (tail_ + 1) % Capacity;
            size_++;
            return true;
        }

        std::optional<T> pop_front() {
            if (size_ == 0) return std::nullopt; // Empty
            T value = std::move(data_[head_]);
            head_ = (head_ + 1) % Capacity;
            size_--;
            return value;
        }

        bool empty() const { return size_ == 0; }
        std::size_t size() const { return size_; }
        std::size_t capacity() const { return Capacity; }

        T& front() {
            if (size_ == 0) throw std::runtime_error("Buffer empty");
            return data_[head_];
        }

        void pop_front_no_return() {
            if (size_ > 0) {
                head_ = (head_ + 1) % Capacity;
                size_--;
            }
        }

        void clear() {
            head_ = 0;
            tail_ = 0;
            size_ = 0;
        }

    private:
        std::array<T, Capacity> data_;
        std::size_t head_;
        std::size_t tail_;
        std::size_t size_;
    };

    /**
     * DynamicRingBuffer: Dynamically resizing FIFO ring buffer / queue allocated with palloc.
     * Replaces std::queue without STL container allocations.
     */
    template <typename T>
    class DynamicRingBuffer {
    public:
        explicit DynamicRingBuffer(size_t initial_cap = 64)
            : cap_(initial_cap < 8 ? 8 : initial_cap),
              head_(0),
              tail_(0),
              size_(0) {
            data_ = static_cast<T*>(palloc_malloc_aligned(cap_ * sizeof(T), alignof(T)));
        }

        ~DynamicRingBuffer() {
            while (size_ > 0) {
                pop();
            }
            if (data_) {
                palloc_free(data_);
                data_ = nullptr;
            }
        }

        DynamicRingBuffer(const DynamicRingBuffer&) = delete;
        DynamicRingBuffer& operator=(const DynamicRingBuffer&) = delete;

        DynamicRingBuffer(DynamicRingBuffer&& other) noexcept
            : data_(other.data_), cap_(other.cap_), head_(other.head_), tail_(other.tail_), size_(other.size_) {
            other.data_ = nullptr;
            other.size_ = 0;
            other.cap_ = 0;
            other.head_ = 0;
            other.tail_ = 0;
        }

        DynamicRingBuffer& operator=(DynamicRingBuffer&& other) noexcept {
            if (this != &other) {
                while (size_ > 0) pop();
                if (data_) palloc_free(data_);
                data_ = other.data_;
                cap_ = other.cap_;
                head_ = other.head_;
                tail_ = other.tail_;
                size_ = other.size_;
                other.data_ = nullptr;
                other.size_ = 0;
                other.cap_ = 0;
                other.head_ = 0;
                other.tail_ = 0;
            }
            return *this;
        }

        bool empty() const noexcept { return size_ == 0; }
        size_t size() const noexcept { return size_; }

        T& front() noexcept {
            return data_[head_];
        }

        const T& front() const noexcept {
            return data_[head_];
        }

        void pop() noexcept {
            if (size_ > 0) {
                data_[head_].~T();
                head_ = (head_ + 1) % cap_;
                --size_;
            }
        }

        template <typename... Args>
        void emplace(Args&&... args) {
            if (size_ == cap_) {
                grow();
            }
            new (&data_[tail_]) T(std::forward<Args>(args)...);
            tail_ = (tail_ + 1) % cap_;
            ++size_;
        }

        void push(const T& val) { emplace(val); }
        void push(T&& val) { emplace(std::move(val)); }

    private:
        void grow() {
            size_t new_cap = cap_ * 2;
            T* new_data = static_cast<T*>(palloc_malloc_aligned(new_cap * sizeof(T), alignof(T)));
            for (size_t i = 0; i < size_; ++i) {
                new (&new_data[i]) T(std::move(data_[(head_ + i) % cap_]));
                data_[(head_ + i) % cap_].~T();
            }
            palloc_free(data_);
            data_ = new_data;
            head_ = 0;
            tail_ = size_;
            cap_ = new_cap;
        }

        T* data_{nullptr};
        size_t cap_{0};
        size_t head_{0};
        size_t tail_{0};
        size_t size_{0};
    };

} // namespace pomai::core::util

namespace pomai::util {
    using pomai::core::util::DynamicRingBuffer;
}

