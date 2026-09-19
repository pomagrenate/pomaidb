#pragma once

// psync_cache_padded.h
// Hardware-sympathetic false-sharing prevention wrapper
// Target: Linux / macOS / Windows across x86-64, ARM64, RISC-V

#include "psync_platform.h"
#include <utility>

namespace psync {

/**
 * CachePadded<T> wraps a type T and pads it to a multiple of the target CPU's
 * destructive interference cache line size (64 bytes on x86-64/ARM Cortex, 128 bytes on Apple Silicon).
 *
 * Prevents "False Sharing" where threads accessing independent adjacent variables
 * thrash the CPU L1/L2 cache coherency protocol.
 */
template <typename T>
class alignas(kDestructiveInterferenceSize) CachePadded {
public:
    T value;

    template <typename... Args>
    constexpr explicit CachePadded(Args&&... args)
        : value(std::forward<Args>(args)...) {}

    constexpr CachePadded() : value() {}

    // Disable implicit copying/moving if T disables it, otherwise support them
    CachePadded(const CachePadded&) = default;
    CachePadded& operator=(const CachePadded&) = default;
    CachePadded(CachePadded&&) noexcept = default;
    CachePadded& operator=(CachePadded&&) noexcept = default;

    constexpr T* operator->() noexcept { return &value; }
    constexpr const T* operator->() const noexcept { return &value; }

    constexpr T& operator*() noexcept { return value; }
    constexpr const T& operator*() const noexcept { return value; }

    constexpr T* get() noexcept { return &value; }
    constexpr const T* get() const noexcept { return &value; }

    constexpr operator T&() noexcept { return value; }
    constexpr operator const T&() const noexcept { return value; }

private:
    // Explicit padding to ensure struct size is a multiple of cache line size
    static constexpr usize kValueSize = sizeof(T);
    static constexpr usize kPaddingBytes = (kValueSize % kDestructiveInterferenceSize == 0)
        ? 0
        : (kDestructiveInterferenceSize - (kValueSize % kDestructiveInterferenceSize));

    // Dummy array when padding is required
    [[maybe_unused]] char padding_[kPaddingBytes > 0 ? kPaddingBytes : 1];
};

} // namespace psync
