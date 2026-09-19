#pragma once

// ptask_types.h
// Type definitions, hardware concurrency detection, thread clamping, and task function wrappers.
// Part of ptask - High-performance work-stealing task scheduler.

#include <cstdint>
#include <cstddef>
#include <thread>
#include <algorithm>
#include <functional>
#include <memory>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(_M_IX86)
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <immintrin.h>
    #endif
#endif

namespace ptask {

// ============================================================================
// CPU PAUSE & YIELD
// ============================================================================

inline void cpu_pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(_M_IX86)
    #if defined(_MSC_VER)
        _mm_pause();
    #else
        __builtin_ia32_pause();
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #if defined(_MSC_VER)
        __yield();
    #else
        asm volatile("yield" ::: "memory");
    #endif
#else
    std::this_thread::yield();
#endif
}

// ============================================================================
// HARDWARE DETECTION & CLAMPING
// ============================================================================

// Returns detected hardware CPU threads (guaranteed >= 1)
inline uint32_t hardware_threads() noexcept {
    uint32_t n = std::thread::hardware_concurrency();
    return n == 0 ? 4 : n;
}

// Clamps user-requested worker thread count to avoid catastrophic CPU over-subscription.
// If requested is 0, defaults to hardware_threads().
// Clamps to [1, max_multiplier * hardware_threads()].
inline uint32_t clamp_workers(uint32_t requested, float max_multiplier = 2.0f) noexcept {
    uint32_t hw = hardware_threads();
    if (requested == 0) {
        return hw;
    }
    uint32_t max_allowed = static_cast<uint32_t>(static_cast<float>(hw) * max_multiplier);
    if (max_allowed < 1) max_allowed = 1;
    return std::clamp(requested, 1u, max_allowed);
}

// ============================================================================
// MOVE-ONLY TYPE-ERASED TASK WRAPPER
// ============================================================================

class Task {
private:
    struct Concept {
        virtual ~Concept() = default;
        virtual void invoke() = 0;
    };

    template <typename F>
    struct Model final : Concept {
        F func;
        explicit Model(F&& f) : func(std::forward<F>(f)) {}
        void invoke() override { func(); }
    };

    std::unique_ptr<Concept> impl_{nullptr};

public:
    Task() noexcept = default;
    
    template <typename F>
    Task(F&& f) : impl_(std::make_unique<Model<std::decay_t<F>>>(std::forward<F>(f))) {}

    Task(Task&&) noexcept = default;
    Task& operator=(Task&&) noexcept = default;

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    void operator()() {
        if (impl_) {
            impl_->invoke();
            impl_.reset();
        }
    }

    explicit operator bool() const noexcept {
        return impl_ != nullptr;
    }

    void reset() noexcept {
        impl_.reset();
    }
};

} // namespace ptask
